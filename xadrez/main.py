import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import chess
import numpy as np
import PySimpleGUI as sg

import os
import math
import time
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor

class Config:
    PROJECT_NAME = "Projeto Quimera"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chess Representation
    NUM_HISTORY_STEPS = 8
    NUM_PIECE_TYPES = 6
    BOARD_DIM = 8
    INPUT_CHANNELS = (NUM_PIECE_TYPES * 2 * NUM_HISTORY_STEPS) + 7
    ACTION_SPACE_SIZE = 4672

    # MCTS
    MCTS_SIMULATIONS = 100
    MCTS_CPUCT = 1.25
    MCTS_TEMPERATURE_START = 1.0
    MCTS_TEMPERATURE_END_MOVE = 30

    # Training
    BATCH_SIZE = 256 
    REPLAY_BUFFER_SIZE = 50000
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    TRAINING_ITERATIONS = 1000
    GAMES_PER_ITERATION = 50 

    # Evaluation
    EVAL_GAMES = 20
    EVAL_WIN_THRESHOLD = 0.55

    # Paths
    MODEL_DIR = "./models/"
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    TEMP_MODEL_PATH = os.path.join(MODEL_DIR, "temp_model.pth")
    
    # GUI
    TOP_K_PREDICTIONS = 5

def uci_to_move_index(uci_move):
    try:
        move = chess.Move.from_uci(uci_move)
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        return abs(hash(uci_move)) % Config.ACTION_SPACE_SIZE
    except:
        return -1

def move_index_to_uci(index, board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return legal_moves[index % len(legal_moves)]


def board_to_tensor(board, history_boards):
    tensor = np.zeros((Config.INPUT_CHANNELS, Config.BOARD_DIM, Config.BOARD_DIM), dtype=np.float32)
    current_player = board.turn
    full_history = history_boards + [board]
    while len(full_history) < Config.NUM_HISTORY_STEPS:
        full_history.insert(0, chess.Board(fen=None))
    for i in range(Config.NUM_HISTORY_STEPS):
        hist_board = full_history[-(i + 1)]
        for square in range(64):
            piece = hist_board.piece_at(square)
            if piece:
                color_offset = 0 if piece.color == current_player else Config.NUM_PIECE_TYPES
                piece_offset = piece.piece_type - 1
                plane = (i * Config.NUM_PIECE_TYPES * 2) + color_offset + piece_offset
                row, col = square // 8, square % 8
                tensor[plane, row, col] = 1

    base_plane_idx = Config.NUM_PIECE_TYPES * 2 * Config.NUM_HISTORY_STEPS
    
    if board.turn == chess.WHITE:
        tensor[base_plane_idx, :, :] = 1
    else:
        tensor[base_plane_idx, :, :] = 0

    tensor[base_plane_idx + 1, :, :] = board.fullmove_number / 100.0
    
    if board.has_kingside_castling_rights(chess.WHITE): tensor[base_plane_idx + 2, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[base_plane_idx + 3, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[base_plane_idx + 4, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[base_plane_idx + 5, :, :] = 1
    tensor[base_plane_idx + 6, :, :] = board.halfmove_clock / 50.0

    return torch.from_numpy(tensor).to(Config.DEVICE)

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class QuimeraNet(nn.Module):
    def __init__(self):
        super(QuimeraNet, self).__init__()
        num_filters = 128
        num_res_blocks = 8 
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(Config.INPUT_CHANNELS, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        self.res_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * Config.BOARD_DIM * Config.BOARD_DIM, Config.ACTION_SPACE_SIZE)
        )
        
        self.opponent_policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * Config.BOARD_DIM * Config.BOARD_DIM, Config.ACTION_SPACE_SIZE)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(Config.BOARD_DIM * Config.BOARD_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.res_tower(out)
        
        pself = self.policy_head(out)
        popp = self.opponent_policy_head(out)
        v = self.value_head(out)
        
        return F.softmax(pself, dim=1), F.softmax(popp, dim=1), v

class MCTSNode:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.q_value = 0
        self.u_value = 0
        self.p_value = prior_p

    def expand(self, action_priors, board):
        legal_moves = {move.uci() for move in board.legal_moves}
        for i, p in enumerate(action_priors):
            action = move_index_to_uci(i, board)
            if action in legal_moves:
                self.children[action] = MCTSNode(parent=self, prior_p=p)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda item: item[1].get_value(c_puct))

    def get_value(self, c_puct):
        self.u_value = c_puct * self.p_value * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q_value + self.u_value

    def update(self, leaf_value):
        self.n_visits += 1
        self.q_value += (leaf_value - self.q_value) / self.n_visits

    def backpropagate(self, leaf_value):
        if self.parent:
            self.parent.backpropagate(-leaf_value)
        self.update(leaf_value)

class MCTS:
    def __init__(self, network):
        self.network = network

    def run(self, board, history_boards):
        root = MCTSNode()
        for _ in range(Config.MCTS_SIMULATIONS):
            node = root
            sim_board = board.copy()
            sim_history = list(history_boards)

            while node.children:
                action, node = node.select(Config.MCTS_CPUCT)
                sim_board.push_uci(action)
                sim_history.append(sim_board.copy())
                if len(sim_history) > Config.NUM_HISTORY_STEPS:
                    sim_history.pop(0)

            if not sim_board.is_game_over():
                state_tensor = board_to_tensor(sim_board, sim_history).unsqueeze(0)
                policy, _, value = self.network(state_tensor)
                policy = policy.squeeze(0).cpu().detach().numpy()
                node.expand(policy, sim_board)
                leaf_value = value.item()
            else:
                outcome = sim_board.outcome()
                if outcome.winner == chess.WHITE:
                    leaf_value = 1 if board.turn == chess.WHITE else -1
                elif outcome.winner == chess.BLACK:
                    leaf_value = -1 if board.turn == chess.WHITE else 1
                else:
                    leaf_value = 0
            
            node.backpropagate(leaf_value)

        return root

    def get_action_probs(self, board, history_boards, temp=1e-3):
        root = self.run(board, history_boards)
        action_visits = {action: node.n_visits for action, node in root.children.items()}
        
        if not action_visits:
             return [], {}
        
        visits = np.array(list(action_visits.values()))
        actions = list(action_visits.keys())
        
        if temp == 0:
            probs = np.zeros_like(visits, dtype=float)
            probs[np.argmax(visits)] = 1.0
        else:
            probs = visits**(1/temp)
            probs /= np.sum(probs)
        
        pi = np.zeros(Config.ACTION_SPACE_SIZE)
        for action, prob in zip(actions, probs):
            idx = uci_to_move_index(action)
            if idx != -1:
                pi[idx] = prob

        return pi, action_visits

class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def add(self, state, pi, value, opp_move):
        self.buffer.append((state, pi, value, opp_move))

def self_play_worker(model_path):
    model = QuimeraNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(Config.DEVICE)
    model.eval()
    
    mcts = MCTS(model)
    
    board = chess.Board()
    history = []
    game_data = []

    while not board.is_game_over():
        temp = Config.MCTS_TEMPERATURE_START if board.fullmove_number < Config.MCTS_TEMPERATURE_END_MOVE else 1e-3
        pi, action_visits = mcts.get_action_probs(board, history, temp)
        
        if not action_visits:
            break

        actions = list(action_visits.keys())
        visits = np.array(list(action_visits.values()))
        chosen_action = random.choices(actions, weights=visits, k=1)[0]
        
        state_tensor = board_to_tensor(board, history)
        
        game_data.append([state_tensor.cpu().numpy(), pi, None, uci_to_move_index(chosen_action)])

        board.push_uci(chosen_action)
        history.append(board.copy())
        if len(history) > Config.NUM_HISTORY_STEPS:
            history.pop(0)

    outcome = board.outcome()
    if outcome:
        if outcome.winner == chess.WHITE: result = 1
        elif outcome.winner == chess.BLACK: result = -1
        else: result = 0
    else:
        result = 0
        
    final_data = []
    player_turn = 1
    for i in range(len(game_data)):
        state, pi, _, opp_move = game_data[i]
        
        value = result if player_turn == 1 else -result
        mask = 1.0
        final_data.append((state, pi, value, opp_move))
        player_turn *= -1

    return final_data

def evaluate_models(model1_path, model2_path):
    
    def play_game(white_model, black_model):
        board = chess.Board()
        history = []
        mcts_white = MCTS(white_model)
        mcts_black = MCTS(black_model)

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                _, action_visits = mcts_white.get_action_probs(board, history, temp=1e-3)
            else:
                _, action_visits = mcts_black.get_action_probs(board, history, temp=1e-3)

            if not action_visits:
                return 0.5 

            chosen_action = max(action_visits, key=action_visits.get)
            board.push_uci(chosen_action)
            history.append(board.copy())
            if len(history) > Config.NUM_HISTORY_STEPS:
                history.pop(0)

        outcome = board.outcome()
        if outcome.winner == chess.WHITE: return 1
        if outcome.winner == chess.BLACK: return -1
        return 0.5 

    model1 = QuimeraNet().to(Config.DEVICE)
    model1.load_state_dict(torch.load(model1_path))
    model1.eval()

    model2 = QuimeraNet().to(Config.DEVICE)
    model2.load_state_dict(torch.load(model2_path))
    model2.eval()

    score = 0
    for i in range(Config.EVAL_GAMES // 2):
        print(f"  Eval game {i*2+1}/{Config.EVAL_GAMES} (New as White)...")
        score += play_game(model1, model2)
        print(f"  Eval game {i*2+2}/{Config.EVAL_GAMES} (New as Black)...")
        score += (1 - play_game(model2, model1))
        
    return score / Config.EVAL_GAMES


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    net = QuimeraNet().to(Config.DEVICE)
    if os.path.exists(Config.BEST_MODEL_PATH):
        net.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
        print("Loaded existing best model.")
    else:
        torch.save(net.state_dict(), Config.BEST_MODEL_PATH)
        print("Initialized new model.")
        
    optimizer = optim.Adam(net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)

    model_promoted = False

    for i in range(Config.TRAINING_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{Config.TRAINING_ITERATIONS} ---")
        
        print("Generating self-play games...")
        net.eval()
        game_data = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self_play_worker, Config.BEST_MODEL_PATH) for _ in range(Config.GAMES_PER_ITERATION)]
            for future in futures:
                game_data.extend(future.result())

        for state, pi, value, opp_move in game_data:
            replay_buffer.add((state, pi, value, opp_move))
        
        print(f"Replay buffer size: {len(replay_buffer)}")
        
        if len(replay_buffer) < Config.BATCH_SIZE:
            print("Buffer not full enough. Skipping training.")
            continue
            
        print("Training network...")
        net.train()
        dataloader = DataLoader(replay_buffer, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        total_loss_val, total_loss_pself, total_loss_popp = 0, 0, 0
        for batch_idx, (states, pis, values, opp_moves) in enumerate(dataloader):
            states = states.to(Config.DEVICE)
            pis = pis.to(Config.DEVICE)
            values = values.to(Config.DEVICE).float().unsqueeze(1)
            opp_moves = opp_moves.to(Config.DEVICE)

            optimizer.zero_grad()
            
            pself_pred, popp_pred, v_pred = net(states)
            
            loss_v = F.mse_loss(v_pred, values)
            loss_pself = -torch.sum(pis * torch.log(pself_pred + 1e-8), dim=1).mean()
            loss_popp = F.cross_entropy(popp_pred, opp_moves)
            total_loss = loss_v + loss_pself + loss_popp
            
            total_loss.backward()
            optimizer.step()

            total_loss_val += loss_v.item()
            total_loss_pself += loss_pself.item()
            total_loss_popp += loss_popp.item()

        print(f"  Losses -> Value: {total_loss_val/len(dataloader):.4f}, Policy: {total_loss_pself/len(dataloader):.4f}, Opponent: {total_loss_popp/len(dataloader):.4f}")

        print("Evaluating new model...")
        torch.save(net.state_dict(), Config.TEMP_MODEL_PATH)
        win_rate = evaluate_models(Config.TEMP_MODEL_PATH, Config.BEST_MODEL_PATH)
        print(f"New model win rate vs best: {win_rate*100:.2f}%")

        if win_rate > Config.EVAL_WIN_THRESHOLD:
            print("!!! NEW BEST MODEL PROMOTED !!!")
            torch.save(net.state_dict(), Config.BEST_MODEL_PATH)
            model_promoted = True
            break
        else:
            print("New model did not meet threshold. Keeping old model.")
            net.load_state_dict(torch.load(Config.BEST_MODEL_PATH))

    if model_promoted:
        print("\nTraining complete. A model was promoted and is ready for play.")
        return True
    else:
        print("\nTraining finished, but no model passed the evaluation threshold.")
        return False

def play_gui():
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("No trained model found. Please run training first.")
        return

    model = QuimeraNet().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    model.eval()
    mcts = MCTS(model)

    board = chess.Board()
    history = []
    
    sg.theme('DarkBlue')

    board_layout = [[sg.Graph((400, 400), (0, 400), (400, 0), key='-BOARD-', change_submits=True, drag_submits=False)]]
    pred_layout = [[sg.Text('AI Predicts You Will Play:', font='Any 14')],
                   [sg.Text('', size=(30,5), key='-PREDICTIONS-', font='Courier 12')]]

    layout = [[sg.Column(board_layout), sg.Column(pred_layout)],
              [sg.Text('Your Move: ', key='-STATUS-'), sg.Button('New Game'), sg.Button('Exit')]]
    
    window = sg.Window(Config.PROJECT_NAME, layout)
    
    def draw_board(graph, board_obj):
        graph.erase()
        for i in range(8):
            for j in range(8):
                color = '#DDB88C' if (i+j)%2==0 else '#A66D4F'
                graph.draw_rectangle((j*50, i*50), (j*50+50, i*50+50), fill_color=color, line_color=color)
        for i in range(64):
            piece = board_obj.piece_at(i)
            if piece:
                filename = f"pieces/{piece.symbol()}.png"
                if os.path.exists(filename):
                     graph.draw_image(filename=filename, location=(i%8*50, i//8*50+50))

    def update_predictions(board_obj, hist_list, model_obj):
        state_tensor = board_to_tensor(board_obj, hist_list).unsqueeze(0)
        _, popp, _ = model_obj(state_tensor)
        popp = popp.squeeze(0).cpu().detach().numpy()
        
        legal_moves_uci = {move.uci() for move in board_obj.legal_moves}
        legal_move_preds = {}

        for move_uci in legal_moves_uci:
            idx = uci_to_move_index(move_uci)
            if idx != -1:
                legal_move_preds[move_uci] = popp[idx]
        
        sorted_preds = sorted(legal_move_preds.items(), key=lambda item: item[1], reverse=True)
        
        pred_text = ""
        for i in range(min(Config.TOP_K_PREDICTIONS, len(sorted_preds))):
            move, prob = sorted_preds[i]
            pred_text += f"{i+1}. {move:<6} ({prob*100:5.2f}%)\n"
        window['-PREDICTIONS-'].update(pred_text)

    # Main GUI loop
    dragged_piece = None
    start_square = None

    if not os.path.exists('pieces'):
        os.makedirs('pieces')
    
    if not os.path.exists('pieces/P.png'):
        import requests, zipfile, io
        print("Downloading chess pieces...")
        try:
            url = "https://github.com/fsmosca/Python-Easy-Chess-GUI/raw/master/pieces.zip"
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()
            print("Pieces downloaded.")
        except Exception as e:
            print(f"Could not download pieces: {e}")
            print("Please download them manually from https://github.com/fsmosca/Python-Easy-Chess-GUI")


    while True:
        event, values = window.read(timeout=100)
        
        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        if event == 'New Game':
            board.reset()
            history.clear()
            window['-STATUS-'].update('Your Move: ')
            
        if not board.is_game_over() and board.turn == chess.WHITE:
             window['-STATUS-'].update('Your Move: ')
             update_predictions(board, history, model)

        draw_board(window['-BOARD-'], board)

        if event == '-BOARD-':
            x, y = values['-BOARD-']
            if x is not None and y is not None:
                col, row = x // 50, y // 50
                square_idx = chess.square(col, 7 - row)
                
                if start_square is None:
                    piece = board.piece_at(square_idx)
                    if piece and piece.color == board.turn:
                        start_square = square_idx
                else:
                    move = chess.Move(start_square, square_idx)
                    if move in board.legal_moves:
                        board.push(move)
                        history.append(board.copy())
                        if len(history) > Config.NUM_HISTORY_STEPS: history.pop(0)
                        draw_board(window['-BOARD-'], board)
                        window.refresh()
                        
                        # AI's turn
                        window['-STATUS-'].update("AI is thinking...")
                        window.refresh()
                        _, action_visits = mcts.get_action_probs(board, history, temp=1e-3)
                        if action_visits:
                            ai_move = max(action_visits, key=action_visits.get)
                            board.push_uci(ai_move)
                            history.append(board.copy())
                            if len(history) > Config.NUM_HISTORY_STEPS: history.pop(0)
                        
                    start_square = None
        
        if board.is_game_over():
            outcome = board.outcome()
            status = "Game Over: "
            if outcome.winner == chess.WHITE: status += "You Win!"
            elif outcome.winner == chess.BLACK: status += "AI Wins!"
            else: status += "Draw!"
            window['-STATUS-'].update(status)

    window.close()


if __name__ == '__main__':
    print(f"Starting {Config.PROJECT_NAME}")
    print(f"Using device: {Config.DEVICE}")

    model_exists = os.path.exists(Config.BEST_MODEL_PATH)
    
    choice = input("Do you want to run training? (y/n): ").lower()
    
    should_play = False
    if choice == 'y':
        should_play = train()
    else:
        if model_exists:
            print("Skipping training. A pre-trained model exists.")
            should_play = True
        else:
            print("Skipping training. No pre-trained model found.")

    if should_play:
        print("\nLaunching GUI to play against the best model...")
        play_gui()
    else:
        print("\nExiting. No model is ready for play.")