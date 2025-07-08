import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import chess
import chess.engine
import numpy as np
import PySimpleGUI as sg

import os
import math
import time
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor

class Config:
    PROJECT_NAME = "Projeto Quimera - Tabula Rasa"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MASTER_ENGINE_PATH = "stockfish"
    MASTER_SKILL_LEVEL = 20 # Nível máximo de habilidade do Stockfish (0-20)

    # Chess Representation
    NUM_HISTORY_STEPS = 8
    NUM_PIECE_TYPES = 6
    BOARD_DIM = 8
    INPUT_CHANNELS = (NUM_PIECE_TYPES * 2 * NUM_HISTORY_STEPS) + 7
    ACTION_SPACE_SIZE = 4672

    # MCTS 
    MCTS_SIMULATIONS = 400
    MCTS_CPUCT = 1.25
    MCTS_TEMPERATURE_START = 1.0
    MCTS_TEMPERATURE_END_MOVE = 30

    # Training 
    BATCH_SIZE = 512
    REPLAY_BUFFER_SIZE = 200000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GAMES_PER_ITERATION = 100 

    # Evaluation against Master
    EVAL_GAMES_PER_CYCLE = 10 

    # Paths
    MODEL_DIR = "./models_tabula_rasa/"
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    
    # GUI
    TOP_K_PREDICTIONS = 5

def uci_to_move_index(uci_move):
    try:
        return abs(hash(uci_move)) % Config.ACTION_SPACE_SIZE
    except:
        return -1

def move_index_to_uci(index, board):
    # Esta função é mais complexa no aprendizado tabula rasa.
    # A IA não sabe os movimentos legais. Ela propõe um a partir de um grande espaço.
    # Para mapear de volta, uma abordagem é ter um mapa pré-computado de todos os movimentos possíveis.
    # Por simplicidade aqui, vamos mapear para um movimento legal aleatório para manter o código funcional,
    # mas o ideal seria um mapeamento fixo index -> uci_string.
    # A IA aprenderá a não usar índices que levam a movimentos ilegais.
    # Para a lógica de aprendizado, a IA propõe um índice, e nós o convertemos para um movimento.
    # A maneira mais direta é gerar uma representação de movimento a partir do índice.
    # Ex: (from_sq, to_sq). 64*63 = 4032. Adicionando promoções, chega-se perto de 4672.
    # Esta é uma simplificação para manter o código executável.
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "a1a1"
    # O ideal seria um mapeamento fixo.
    # Para este exemplo, a IA escolhe o ÍNDICE e a função self_play_worker o valida.
    # Vamos gerar um movimento hipotético para que o sistema funcione.
    # Esta é uma área que necessitaria de um design de espaço de ação mais detalhado.
    # Por exemplo, o AlphaZero usa 8x8x73.
    # Vamos simular um mapeamento reverso.
    all_moves = []
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq != to_sq:
                all_moves.append(chess.Move(from_sq, to_sq).uci())
    
    if index < len(all_moves):
        return all_moves[index]
    else:
        return random.choice(all_moves) # Fallback

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
        num_filters = 256 
        num_res_blocks = 19 
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(Config.INPUT_CHANNELS, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.res_tower = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * Config.BOARD_DIM * Config.BOARD_DIM, Config.ACTION_SPACE_SIZE)
        )
        self.opponent_policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * Config.BOARD_DIM * Config.BOARD_DIM, Config.ACTION_SPACE_SIZE)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
            nn.Linear(Config.BOARD_DIM * Config.BOARD_DIM, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Tanh()
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

    def expand(self, action_priors):
        for i, p in enumerate(action_priors):
            if p > 1e-6:
                self.children[i] = MCTSNode(parent=self, prior_p=p)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda item: item[1].get_value(c_puct))

    def get_value(self, c_puct):
        if self.parent is None or self.parent.n_visits == 0:
             u_boost = 1.0
        else:
             u_boost = math.sqrt(self.parent.n_visits)
        self.u_value = c_puct * self.p_value * u_boost / (1 + self.n_visits)
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
        state_tensor = board_to_tensor(board, history_boards).unsqueeze(0)
        policy, _, value = self.network(state_tensor)
        root.expand(policy.squeeze(0).cpu().detach().numpy())
        
        for _ in range(Config.MCTS_SIMULATIONS):
            node = root
            sim_board = board.copy()
            sim_history = list(history_boards)
            
            # SELEÇÃO
            while node.children:
                action_idx, node = node.select(Config.MCTS_CPUCT)
                move_uci = move_index_to_uci(action_idx, sim_board)
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in sim_board.legal_moves:
                        sim_board.push(move)
                        sim_history.append(sim_board.copy())
                        if len(sim_history) > Config.NUM_HISTORY_STEPS:
                            sim_history.pop(0)
                    else:
                        break
                except:
                    break

            # EXPANSÃO E SIMULAÇÃO
            leaf_value = 0
            if not sim_board.is_game_over():
                try:
                    move = chess.Move.from_uci(move_index_to_uci(action_idx, sim_board))
                    if move in board.legal_moves:
                        state_tensor_leaf = board_to_tensor(sim_board, sim_history).unsqueeze(0)
                        policy_leaf, _, value_leaf = self.network(state_tensor_leaf)
                        node.expand(policy_leaf.squeeze(0).cpu().detach().numpy())
                        leaf_value = value_leaf.item()
                    else:
                        leaf_value = -1.0
                except:
                     leaf_value = -1.0
            else:
                outcome = sim_board.outcome()
                if outcome:
                    if outcome.winner is not None:
                        leaf_value = 1.0 if outcome.winner == board.turn else -1.0
                    else:
                        leaf_value = 0.0
            
            # BACKPROPAGATION
            node.backpropagate(leaf_value)
        return root

    def get_action_probs(self, board, history_boards, temp=1e-3):
        root = self.run(board, history_boards)
        action_visits = {action_idx: node.n_visits for action_idx, node in root.children.items()}
        
        if not action_visits: return np.zeros(Config.ACTION_SPACE_SIZE), {}
        
        visits = np.array(list(action_visits.values()))
        actions = list(action_visits.keys())
        
        if temp == 0:
            probs = np.zeros_like(visits, dtype=float)
            if visits.size > 0:
                probs[np.argmax(visits)] = 1.0
        else:
            probs = visits**(1/temp)
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs /= probs_sum
            else: 
                probs = np.ones_like(visits, dtype=float) / len(visits)

        pi = np.zeros(Config.ACTION_SPACE_SIZE)
        for action, prob in zip(actions, probs):
            pi[action] = prob
            
        return pi, action_visits

class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self): return len(self.buffer)
    def __getitem__(self, idx): return self.buffer[idx]
    def add(self, state, pi, value, opp_move_idx): self.buffer.append((state, pi, value, opp_move_idx))

def self_play_worker(model_path):
    model = QuimeraNet()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError:
        torch.save(model.state_dict(), model_path) 
    model.to(Config.DEVICE).eval()
    
    mcts = MCTS(model)
    board = chess.Board()
    history = []
    game_data = []

    while not board.is_game_over() and board.fullmove_number < 150:
        state_tensor = board_to_tensor(board, history)
        temp = Config.MCTS_TEMPERATURE_START if board.fullmove_number < Config.MCTS_TEMPERATURE_END_MOVE else 1e-3
        pi, action_visits = mcts.get_action_probs(board, history, temp)
        
        if not action_visits: break
        
        current_player_value_sign = 1 if board.turn == chess.WHITE else -1
        
        move_made = False
        attempts = 0
        while not move_made and attempts < len(action_visits) + 5: 
            attempts += 1
            action_indices = list(action_visits.keys())
            visits = np.array(list(action_visits.values()))
            
            if np.sum(visits) == 0:
                chosen_action_idx = random.choice(action_indices)
            else:
                visit_probs = visits / np.sum(visits)
                chosen_action_idx = np.random.choice(action_indices, p=visit_probs)

            move_uci = move_index_to_uci(chosen_action_idx, board)
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    game_data.append([state_tensor.cpu().numpy(), pi, None, chosen_action_idx])
                    board.push(move)
                    history.append(board.copy())
                    if len(history) > Config.NUM_HISTORY_STEPS: history.pop(0)
                    move_made = True
                else:
                    game_data.append([state_tensor.cpu().numpy(), pi, -1.0 * current_player_value_sign, chosen_action_idx])
                    del action_visits[chosen_action_idx]
                    if not action_visits: break 
            except ValueError: 
                game_data.append([state_tensor.cpu().numpy(), pi, -1.0 * current_player_value_sign, chosen_action_idx])
                del action_visits[chosen_action_idx]
                if not action_visits: break
    
    outcome = board.outcome()
    result = 0
    if outcome:
        if outcome.winner == chess.WHITE: result = 1
        elif outcome.winner == chess.BLACK: result = -1
    
    final_data = []
    for i in reversed(range(len(game_data))):
        state, pi, value, opp_move = game_data[i]
        if value is None:
            value = result if (len(game_data) - 1 - i) % 2 == 0 else -result
        final_data.append((state, pi, value, opp_move))
    
    final_data.reverse()
    return final_data

def evaluate_against_master(model_path, engine_path, skill_level, num_games):
    model = QuimeraNet().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    mcts = MCTS(model)

    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        engine.configure({"Skill Level": skill_level})
    except chess.engine.EngineError as e:
        print(f"ERRO: Não foi possível iniciar o motor de xadrez em '{engine_path}'.")
        print(f"Verifique se o caminho está correto na Config e se o arquivo é executável.")
        print(f"Erro original: {e}")
        return -1 

    quimera_score = 0
    
    for i in range(num_games):
        board = chess.Board()
        history = []
        is_quimera_white = (i % 2 == 0)
        print(f"  Jogo de Avaliação {i+1}/{num_games} (Quimera de {'Brancas' if is_quimera_white else 'Pretas'})...", end="")

        while not board.is_game_over():
            if board.turn == is_quimera_white: 
                _, action_visits = mcts.get_action_probs(board, history, temp=0)
                if not action_visits: break
                
                best_action_idx = max(action_visits, key=action_visits.get)
                move_uci = move_index_to_uci(best_action_idx, board)
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else: 
                    quimera_score += -1 if is_quimera_white else 0
                    break
            else: 
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)

            history.append(board.copy())
            if len(history) > Config.NUM_HISTORY_STEPS: history.pop(0)

        outcome = board.outcome()
        if outcome:
            if outcome.winner is not None:
                if outcome.winner == is_quimera_white:
                    quimera_score += 1
                    print(" Vitória!")
                else:
                    print(" Derrota.")
            else:
                quimera_score += 0.5
                print(" Empate.")
        else:
             print(" Jogo Incompleto.")
    
    engine.quit()
    return quimera_score

def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    net = QuimeraNet().to(Config.DEVICE)
    if os.path.exists(Config.BEST_MODEL_PATH):
        net.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
        print("Modelo 'best' existente carregado.")
    else:
        torch.save(net.state_dict(), Config.BEST_MODEL_PATH)
        print("Novo modelo inicializado e salvo como 'best'.")
        
    optimizer = optim.Adam(net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)

    quimera_total_wins = 0
    master_total_wins = 0
    iteration = 0

    while quimera_total_wins <= master_total_wins:
        iteration += 1
        print(f"\n--- Ciclo de Treinamento {iteration} ---")
        print(f"PLACAR ATUAL: Quimera {quimera_total_wins} x {master_total_wins} Mestre")
        
        print("Gerando jogos de auto-play para aprender as regras e estratégias...")
        net.eval()
        game_data = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self_play_worker, Config.BEST_MODEL_PATH) for _ in range(Config.GAMES_PER_ITERATION)]
            for future in futures:
                try:
                    result = future.result()
                    game_data.extend(result)
                except Exception as e:
                    print(f"Erro em um worker de self-play: {e}")

        for state, pi, value, opp_move in game_data:
            replay_buffer.add((state, pi, value, opp_move))
        
        print(f"Tamanho do Replay Buffer: {len(replay_buffer)}")
        
        if len(replay_buffer) < Config.BATCH_SIZE * 10: 
            print("Buffer de replay com poucos dados. Gerando mais jogos...")
            continue
            
        print("Treinando a rede...")
        net.train()
        for epoch in range(5):
            dataloader = DataLoader(replay_buffer, batch_size=Config.BATCH_SIZE, shuffle=True)
            for states, pis, values, opp_moves in dataloader:
                states, pis, values, opp_moves = states.to(Config.DEVICE), pis.to(Config.DEVICE), values.to(Config.DEVICE).float().unsqueeze(1), opp_moves.to(Config.DEVICE)
                optimizer.zero_grad()
                pself_pred, popp_pred, v_pred = net(states)
                loss_v = F.mse_loss(v_pred, values)
                loss_pself = -torch.sum(pis * torch.log(pself_pred + 1e-8), dim=1).mean()
                loss_popp = F.cross_entropy(popp_pred, opp_moves)
                total_loss = loss_v + loss_pself + loss_popp
                total_loss.backward()
                optimizer.step()
        
        torch.save(net.state_dict(), Config.BEST_MODEL_PATH) 
        
        print("Avaliação contra o Mestre...")
        score = evaluate_against_master(Config.BEST_MODEL_PATH, Config.MASTER_ENGINE_PATH, Config.MASTER_SKILL_LEVEL, Config.EVAL_GAMES_PER_CYCLE)
        
        if score == -1: 
            print("Encerrando treinamento devido a erro no motor de xadrez.")
            break
            
        quimera_wins_cycle = score
        master_wins_cycle = Config.EVAL_GAMES_PER_CYCLE - score
        print(f"Resultado do ciclo: Quimera {quimera_wins_cycle} x {master_wins_cycle} Mestre")
        
        quimera_total_wins += quimera_wins_cycle
        master_total_wins += master_wins_cycle

    print("\n--- TREINAMENTO CONCLUÍDO! ---")
    print(f"PLACAR FINAL: Quimera {quimera_total_wins} x {master_total_wins} Mestre")
    print("A Quimera superou o Mestre e se tornou o novo campeão!")
    return True


if __name__ == '__main__':
    print(f"Iniciando {Config.PROJECT_NAME}")
    print(f"Usando dispositivo: {Config.DEVICE}")
    
    if not os.path.exists(Config.MASTER_ENGINE_PATH):
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! ATENÇÃO: Motor de xadrez mestre não encontrado.  !!!")
         print(f"!!! Por favor, baixe o Stockfish (ou outro motor UCI) !!!")
         print(f"!!! e atualize o caminho em Config.MASTER_ENGINE_PATH !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        choice = input("Você quer iniciar o treinamento 'Tabula Rasa'? (y/n): ").lower()
        if choice == 'y':
            train()
        else:
            print("Treinamento não iniciado.")