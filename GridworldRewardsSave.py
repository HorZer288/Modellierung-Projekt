import numpy as np
import pygame
import random # Für die Epsilon-Greedy-Strategie
import os # Für Dateipfade

class Gridworld:
    """
    Repräsentiert die Gridworld-Umgebung für das Reinforcement Learning.
    Beinhaltet die Gitterstruktur, Start-/Ziel-/Fallen-/Belohnungspositionen
    und die Logik für Zustandsübergänge und Belohnungen.
    """
    def __init__(self, width, height, start_pos, goal_pos_list, num_rewards, num_traps,
                 reward_goal=100, reward_regular_step=-1, reward_trap=-20, reward_collect=50,
                 initial_agent_points=0):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start_pos = start_pos
        self.goal_pos_list = goal_pos_list
        self.num_rewards = num_rewards
        self.num_traps = num_traps

        # NEU: Parameter für die erweiterte Zustandsdefinition
        # Jede Nachbarzelle kann 3 Typen haben: 0 (leer/Start/Ziel), 1 (Belohnung), 2 (Falle)
        self.NUM_NEIGHBOR_TYPES = 3 
        # Der Status der gesammelten Belohnungen wird NICHT mehr in den Zustand einkodiert,
        # um den Zustandsraum zu reduzieren und die Trainingszeit zu verkürzen.
        # Der Agent nimmt ein Belohnungsfeld, nachdem es eingesammelt wurde, als 'EMPTY' wahr.

        # Berechnung der Gesamtzahl der Zustände mit vereinfachter Wahrnehmung
        # Zustand = (Agentenposition) * (Nachbarfeld-Konfiguration)
        self.num_states = (self.width * self.height) * \
                          (self.NUM_NEIGHBOR_TYPES ** 4) 

        # Zellen-Typen (Konstanten)
        self.EMPTY = 0
        self.START = 1
        self.GOAL = 2
        self.TRAP = 3
        self.REWARD = 4

        # Aktionen (Konstanten)
        self.ACTIONS = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3
        }
        self.num_actions = len(self.ACTIONS)

        # Belohnungswerte (konfigurierbar über den Konstruktor)
        self.REWARD_GOAL = reward_goal
        self.REWARD_REGULAR_STEP = reward_regular_step
        self.REWARD_TRAP = reward_trap
        self.REWARD_COLLECT = reward_collect
        self.INITIAL_AGENT_POINTS = initial_agent_points

        # FIX: _initialize_grid muss einmal aufgerufen werden, um _base_grid_state zu setzen
        self._initialize_grid() 
        
        # NEU: Diese Listen werden nicht mehr für die Zustands-Einkodierung benötigt,
        # da der Sammelstatus nicht mehr Teil des Zustands ist.
        # Sie können entfernt werden, da die Gitterzelle direkt auf EMPTY gesetzt wird.
        # self.current_reward_positions = [] 
        # self.rewards_collected_status = [False] * self.num_rewards 

        self.reset() 


    def _initialize_grid(self):
        """
        Initialisiert das Gitter mit Start und Zielen.
        Speichert den Grundzustand (ohne zufällige Elemente) für spätere Resets.
        """
        self.grid.fill(self.EMPTY)

        # Startposition setzen
        self.grid[self.start_pos[1], self.start_pos[0]] = self.START

        # Zielposition(en) setzen
        for gx, gy in self.goal_pos_list:
            self.grid[gy, gx] = self.GOAL
        
        # Speichere diesen Grundzustand (ohne zufällige Elemente)
        # Dies ist der Zustand, auf den wir zurücksetzen, bevor wir Fallen/Belohnungen neu platzieren.
        self._base_grid_state = np.copy(self.grid)


    def reset(self):
        """
        Setzt die Umgebung auf einen neuen, zufälligen Zustand zurück.
        Belohnungsfelder und Fallen werden neu platziert.
        Gibt den Startzustand und die initialen Punkte des Agenten zurück.
        """
        # Stelle das Gitter auf den initialen Grundzustand zurück (nur Start/Ziel)
        self.grid = np.copy(self._base_grid_state)

        # NEU: Belohnungs-Sammelstatus und Positionen werden nicht mehr zurückgesetzt,
        # da sie nicht mehr Teil der Zustandsdefinition sind.
        # self.rewards_collected_status = [False] * self.num_rewards
        # self.current_reward_positions = [] 

        # Zufällige Platzierung von Belohnungen und Fallen bei jedem Reset
        possible_positions = []
        for y in range(self.height):
            for x in range(self.width):
                # Stellen sicher, dass Belohnungen/Fallen nicht auf Start/Ziel liegen
                if (x, y) != self.start_pos and (x, y) not in self.goal_pos_list:
                    possible_positions.append((x, y))

        np.random.shuffle(possible_positions) # Zufällige Reihenfolge

        # Fallen platzieren
        for _ in range(self.num_traps):
            if not possible_positions: # falls keine Plätze mehr frei sind
                break
            x, y = possible_positions.pop(0)
            self.grid[y, x] = self.TRAP

        # Belohnungen platzieren
        for i in range(self.num_rewards): 
            if not possible_positions: # falls keine Plätze mehr frei sind
                break
            x, y = possible_positions.pop(0)
            self.grid[y, x] = self.REWARD
            # NEU: Position der Belohnung wird nicht mehr explizit gespeichert,
            # da der Sammelstatus nicht mehr Teil des Zustands ist.
            # self.current_reward_positions.append((x, y)) 


        # NEU: Rückgabe des erweiterten Startzustands (ohne Belohnungs-Sammelstatus)
        return self.get_state(self.start_pos[0], self.start_pos[1]), self.INITIAL_AGENT_POINTS

    # NEU: Hilfsfunktion zur Kodierung des Zelltyps für den Zustand
    def _get_cell_type_for_state_encoding(self, x, y):
        if not self.is_valid_coord(x, y):
            return 0 # Außerhalb der Grenzen als leer behandeln
        cell_type = self.grid[y, x]
        if cell_type == self.REWARD:
            return 1
        elif cell_type == self.TRAP:
            return 2
        else: # EMPTY, START, GOAL
            return 0

    def get_state(self, agent_x, agent_y):
        """
        Konvertiert Agenten-Koordinaten und Umgebungsdetails
        in einen einzelnen Zustands-Index.
        Der Zustand beinhaltet jetzt die Agentenposition und die Typen der 4 Nachbarfelder.
        """
        # 1. Agentenposition kodieren (0 bis width*height - 1)
        pos_state = agent_y * self.width + agent_x

        # 2. Nachbarfelder kodieren (Reihenfolge: Oben, Unten, Links, Rechts)
        neighbor_types = [
            self._get_cell_type_for_state_encoding(agent_x, agent_y - 1), # Oben
            self._get_cell_type_for_state_encoding(agent_x, agent_y + 1), # Unten
            self._get_cell_type_for_state_encoding(agent_x - 1, agent_y), # Links
            self._get_cell_type_for_state_encoding(agent_x + 1, agent_y)  # Rechts
        ]

        neighbor_encoding = 0
        for i, cell_type in enumerate(neighbor_types):
            neighbor_encoding += cell_type * (self.NUM_NEIGHBOR_TYPES ** i)
        
        # NEU: Belohnungs-Sammelstatus wird NICHT mehr kodiert.
        # Der finale Zustand ist eine Kombination aus Position und Nachbar-Kodierung
        # state = pos_state + \
        #         (neighbor_encoding * (self.width * self.height)) + \
        #         (reward_status_encoding * (self.width * self.height) * (self.NUM_NEIGHBOR_TYPES ** 4))
        state = pos_state + \
                (neighbor_encoding * (self.width * self.height)) 
        
        return state

    def get_coords_from_state(self, state_index):
        """
        Konvertiert einen Zustands-Index zurück in Agenten-Koordinaten (x, y).
        Diese Funktion extrahiert nur die Position aus dem erweiterten Zustand.
        """
        # Zuerst den Positions-Anteil des Zustands extrahieren
        pos_state = state_index % (self.width * self.height)
        
        y = pos_state // self.width
        x = pos_state % self.width
        return x, y

    def is_valid_coord(self, x, y):
        """Prüft, ob Koordinaten (x, y) innerhalb der Gridworld liegen."""
        return 0 <= x < self.width and 0 <= y < self.height

    def step(self, current_state_index, action):
        """
        Führt eine Aktion in der Gridworld aus und gibt den neuen Zustand,
        die erhaltene Belohnung und ob der Zustand terminal ist, zurück.
        """
        # Zuerst die Agentenposition aus dem erweiterten Zustand extrahieren
        current_x, current_y = self.get_coords_from_state(current_state_index)
        new_x, new_y = current_x, current_y
        
        reward = self.REWARD_REGULAR_STEP # Standardbelohnung (Kosten pro Schritt)
        is_terminal = False

        # Neue Koordinaten basierend auf der Aktion berechnen
        if action == self.ACTIONS["UP"]:
            new_y -= 1
        elif action == self.ACTIONS["DOWN"]:
            new_y += 1
        elif action == self.ACTIONS["LEFT"]:
            new_x -= 1
        elif action == self.ACTIONS["RIGHT"]:
            new_x += 1

        # Prüfen, ob die neue Position gültig ist (innerhalb der Grenzen)
        if not self.is_valid_coord(new_x, new_y):
            # Agent versucht, aus der Welt zu laufen -> Bleibt an aktueller Position
            new_x, new_y = current_x, current_y 

        # Typ der Zelle an der neuen Position bestimmen
        cell_type_at_new_pos = self.grid[new_y, new_x]

        # Belohnung anpassen und Terminalstatus setzen basierend auf dem Zelltyp
        if cell_type_at_new_pos == self.GOAL:
            reward += self.REWARD_GOAL # Belohnung für Ziel
            is_terminal = True         # Ziel ist ein Endzustand
        elif cell_type_at_new_pos == self.TRAP:
            reward += self.REWARD_TRAP # Punkteabzug für Falle
        elif cell_type_at_new_pos == self.REWARD:
            reward += self.REWARD_COLLECT # Punktegutschrift für Belohnung
            # Belohnung nur einmal pro Episode einsammeln: Feld in leer umwandeln
            self.grid[new_y, new_x] = self.EMPTY 
            
            # NEU: Status der gesammelten Belohnung wird nicht mehr explizit aktualisiert,
            # da er nicht mehr Teil des Zustands ist.
            # if (new_x, new_y) in self.current_reward_positions:
            #     idx = self.current_reward_positions.index((new_x, new_y))
            #     self.rewards_collected_status[idx] = True


        # NEU: Der nächste Zustand wird mit der vereinfachten Zustandsdefinition berechnet
        new_state_index = self.get_state(new_x, new_y)
        
        return new_state_index, reward, is_terminal

    def display(self, agent_pos=None):
        """
        Stellt die Gridworld als ASCII dar (optional, kann beibehalten werden).
        Nützlich für schnelles Debugging ohne Pygame-Fenster.
        """
        display_grid = np.copy(self.grid).astype(str)
        display_grid[display_grid == str(self.EMPTY)] = '.'
        display_grid[display_grid == str(self.START)] = 'S'
        display_grid[display_grid == str(self.GOAL)] = 'G'
        display_grid[display_grid == str(self.TRAP)] = 'T'
        display_grid[display_grid == str(self.REWARD)] = 'R'

        if agent_pos:
            ax, ay = agent_pos
            display_grid[ay, ax] = 'A' # Agent

        print("-" * (self.width * 2 + 1))
        for row in display_grid:
            print("|" + "|".join(row) + "|")
        print("-" * (self.width * 2 + 1))


class GridworldVisualizer:
    """
    Kümmert sich um die grafische Darstellung der Gridworld mit Pygame.
    """
    def __init__(self, gridworld, cell_size=60):
        self.gridworld = gridworld
        self.cell_size = cell_size
        self.width_px = gridworld.width * cell_size
        self.height_px = gridworld.height * cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width_px, self.height_px))
        pygame.display.set_caption("Gridworld Reinforcement Learning")
        self.clock = pygame.time.Clock()

        # Farben definieren (RGB)
        self.COLORS = {
            self.gridworld.EMPTY: (255, 255, 255),  # Weiß
            self.gridworld.START: (0, 150, 0),    # Dunkelgrün
            self.gridworld.GOAL: (0, 0, 255),     # Blau
            self.gridworld.TRAP: (255, 0, 0),     # Rot
            self.gridworld.REWARD: (255, 255, 0),  # Gelb
            "AGENT": (0, 0, 0),                   # Schwarz
            "GRID_LINES": (150, 150, 150)         # Grau
        }
        self.font = pygame.font.Font(None, 30) # Schrift für Text in Zellen

    def draw_grid(self, grid_data, agent_pos=None, episode_info=None, current_score=None): # NEU: current_score hinzugefügt
        """
        Zeichnet die Gridworld und den Agenten.
        grid_data: Die spezifische Gitterkonfiguration, die gezeichnet werden soll.
        Optional können Episode-Informationen und die aktuelle Punktzahl angezeigt werden.
        """
        self.screen.fill(self.COLORS[self.gridworld.EMPTY]) # Hintergrund füllen

        for y in range(self.gridworld.height):
            for x in range(self.gridworld.width):
                # Verwende grid_data statt self.gridworld.grid
                cell_type = grid_data[y, x] 
                color = self.COLORS[cell_type]
                
                # Rechteck für die Zelle zeichnen
                pygame.draw.rect(self.screen, color, 
                                 (x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size))
                
                # Gitterlinien zeichnen
                pygame.draw.rect(self.screen, self.COLORS["GRID_LINES"], 
                                 (x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size), 1) # 1 ist die Linienbreite

                # Text für S, G, T, R hinzufügen
                text_surface = None
                if cell_type == self.gridworld.START:
                    text_surface = self.font.render("S", True, (255,255,255))
                elif cell_type == self.gridworld.GOAL:
                    text_surface = self.font.render("G", True, (255,255,255))
                elif cell_type == self.gridworld.TRAP:
                    text_surface = self.font.render("T", True, (255,255,255))
                elif cell_type == self.gridworld.REWARD:
                    text_surface = self.font.render("R", True, (0,0,0))
                
                if text_surface:
                    text_rect = text_surface.get_rect(center=(x * self.cell_size + self.cell_size // 2, 
                                                               y * self.cell_size + self.cell_size // 2))
                    self.screen.blit(text_surface, text_rect)

        # Agent zeichnen, wenn Position gegeben
        if agent_pos:
            ax, ay = agent_pos
            center_x = ax * self.cell_size + self.cell_size // 2
            center_y = ay * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.COLORS["AGENT"], (center_x, center_y), self.cell_size // 3)

        # Episode-Informationen anzeigen
        if episode_info:
            info_text_str = episode_info
            if current_score is not None: # NEU: Punktzahl zur Anzeige hinzufügen
                info_text_str += f" | Punkte: {current_score}"
            info_text = self.font.render(info_text_str, True, (0, 0, 0)) # Schwarzer Text
            self.screen.blit(info_text, (10, 10)) # Oben links

        pygame.display.flip() # Alles auf dem Bildschirm aktualisieren

    def run_visualization(self, grid_data_at_episode_start, agent_pos_history=None, score_history=None, fps=10): # NEU: score_history hinzugefügt
        """
        Spielt die Visualisierung des Agentenpfades ab.
        grid_data_at_episode_start: Der Zustand des Gitters zu Beginn der Episode.
        agent_pos_history: Liste der (x,y) Positionen des Agenten.
        score_history: Liste der Punktzahlen des Agenten bei jedem Schritt.
        fps: Bilder pro Sekunde für die Animation.
        """
        running = True
        current_step_idx = 0
        num_steps = len(agent_pos_history) if agent_pos_history else 0
        
        # Initialisiere die Visualisierung mit der ersten Position und Punktzahl
        if num_steps > 0:
            current_agent_pos = agent_pos_history[0]
            current_score = score_history[0] if score_history else None # NEU: Initialer Score
        else:
            current_agent_pos = None
            current_score = None

        auto_advance = True # Startet die Animation automatisch

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: # Leertaste für nächsten Schritt (manuell)
                        auto_advance = False # Schaltet automatischen Vorschub aus
                        if num_steps > 0:
                            current_step_idx = (current_step_idx + 1) % num_steps
                            current_agent_pos = agent_pos_history[current_step_idx]
                            current_score = score_history[current_step_idx] if score_history else None # NEU: Score aktualisieren
                    elif event.key == pygame.K_RETURN: # Enter-Taste für automatischen Vorschub
                        auto_advance = True
                    elif event.key == pygame.K_ESCAPE: # ESC zum Beenden der Visualisierung
                        running = False
            
            if auto_advance and num_steps > 0:
                # Automatischer Vorschub des Agenten
                current_step_idx = (current_step_idx + 1) % num_steps
                current_agent_pos = agent_pos_history[current_step_idx]
                current_score = score_history[current_step_idx] if score_history else None # NEU: Score aktualisieren

            # Übergebe grid_data_at_episode_start und current_score an draw_grid
            self.draw_grid(grid_data=grid_data_at_episode_start, 
                           agent_pos=current_agent_pos, 
                           episode_info=f"Schritt: {current_step_idx+1}/{num_steps}",
                           current_score=current_score) # NEU: current_score übergeben
            
            self.clock.tick(fps) # Begrenzt die Bildrate

        pygame.quit()


class QLearningAgent:
    """
    Implementiert den Q-Learning-Algorithmus für einen Agenten.
    """
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate    # Alpha (α)
        self.discount_factor = discount_factor # Gamma (γ)
        self.epsilon = epsilon                # Epsilon (ε) für Exploration
        self.epsilon_decay_rate = epsilon_decay_rate # Rate, mit der Epsilon abnimmt
        self.min_epsilon = min_epsilon        # Minimaler Epsilon-Wert

        # Q-Tabelle initialisieren: Zeilen = Zustände, Spalten = Aktionen
        # Alle Werte starten bei 0
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state):
        """
        Wählt eine Aktion basierend auf der Epsilon-Greedy-Strategie.
        Mit Wahrscheinlichkeit Epsilon: zufällige Aktion (Exploration)
        Sonst: beste Aktion aus Q-Tabelle (Exploitation)
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Wähle eine zufällige Aktion
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploitation: Wähle die Aktion mit dem höchsten Q-Wert für den aktuellen Zustand
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, is_terminal):
        """
        Aktualisiert den Q-Wert in der Q-Tabelle basierend auf der Q-Learning-Formel.
        Q(S, A) <- Q(S, A) + α * [R + γ * max(Q(S', A')) - Q(S, A)]
        """
        current_q_value = self.q_table[state, action]

        if is_terminal:
            # Wenn der nächste Zustand terminal ist, gibt es keine zukünftigen Belohnungen
            max_future_q = 0
        else:
            # Finde den maximalen Q-Wert für den nächsten Zustand (S') über alle Aktionen (A')
            max_future_q = np.max(self.q_table[next_state, :])

        # Q-Learning-Formel anwenden
        new_q_value = current_q_value + self.learning_rate * \
                      (reward + self.discount_factor * max_future_q - current_q_value)
        
        self.q_table[state, action] = new_q_value

    def decay_epsilon(self):
        """
        Reduziert den Epsilon-Wert über die Zeit, um von Exploration zu Exploitation zu wechseln.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def get_policy(self, env):
        """
        Gibt die gelernte optimale Strategie (Policy) als Liste von Aktionen für jeden Zustand zurück.
        """
        policy = []
        for state in range(self.num_states):
            # Wähle die Aktion mit dem höchsten Q-Wert für jeden Zustand
            action_index = np.argmax(self.q_table[state, :])
            # Konvertiere den numerischen Aktionsindex zurück in den Aktionsnamen
            action_name = list(env.ACTIONS.keys())[list(env.ACTIONS.values()).index(action_index)]
            policy.append(action_name)
        return policy

    def save_q_table(self, filename="q_table.npy"):
        """
        Speichert die Q-Tabelle in einer .npy Datei.
        """
        np.save(filename, self.q_table)
        print(f"Q-Tabelle erfolgreich unter '{filename}' gespeichert.")

    def load_q_table(self, filename="q_table.npy"):
        """
        Lädt die Q-Tabelle aus einer .npy Datei.
        Gibt True zurück, wenn erfolgreich, False sonst.
        """
        if os.path.exists(filename):
            self.q_table = np.load(filename)
            print(f"Q-Tabelle erfolgreich von '{filename}' geladen.")
            return True
        else:
            print(f"Fehler: Datei '{filename}' nicht gefunden. Kann Q-Tabelle nicht laden.")
            return False


# --- Hauptteil des Skripts: Konfiguration und Trainingsablauf ---

if __name__ == "__main__":
    # --- Konfigurierbare Parameter für Gridworld ---
    GRID_WIDTH = 5 
    GRID_HEIGHT = 5  
    START_COORDINATES = (0, GRID_HEIGHT - 1) 
    GOAL_COORDINATES = [(GRID_WIDTH - 1, 0)] 
    NUM_REWARDS_ON_MAP = 4 
    NUM_TRAPS_ON_MAP = 2 
    
    # Belohnungswerte 
    REWARD_GOAL_VALUE = 1500      # Belohnung für das Ziel 
    REWARD_REGULAR_STEP_VALUE = -5 # Schrittkosten
    REWARD_TRAP_VALUE = -10000     # Kosten für Fallen
    REWARD_COLLECT_VALUE = 1000    # Belohnungspunkte
    INITIAL_AGENT_POINTS = 0       # Startpunkte

    # --- Parameter für Q-Learning Agent ---
    LEARNING_RATE = 0.1            # Alpha (α)
    DISCOUNT_FACTOR = 0.99         # Gamma (γ) 
    EPSILON_START = 1.0            # Startwert für Epsilon (100% Exploration)
    MIN_EPSILON = 0.01             # Minimaler Epsilon-Wert
    

    NUM_EPISODES = 2000000         # Anzahl an Durchläufen
    EPSILON_DECAY_RATE = (EPSILON_START - MIN_EPSILON) / NUM_EPISODES 
    
    MAX_STEPS_PER_EPISODE = 500   # Maximale Schritte pro Episode
    
    
    MIN_EPSILON_EVALUATION = 0.0 # Keine Exploration in der Evaluierung

    # Visualisierungseinstellungen
    VISUALIZE_EVERY_N_EPISODES = 2000000 # Alle X Episoden visualisieren 
    VISUALIZATION_FPS = 15           # Bilder pro Sekunde für die Visualisierung
    NUM_EVAL_EPISODES = 10           # Anzahl der Episoden für die Evaluierung nach dem Training

    # Dateiname für die Q-Tabelle
    Q_TABLE_FILENAME = "trained_q_table75.npy"

    # --- Initialisierung der Umgebung und des Agenten ---
    # Die Gridworld-Initialisierung muss zuerst erfolgen, um env.num_states korrekt zu setzen
    env = Gridworld(GRID_WIDTH, GRID_HEIGHT, START_COORDINATES, GOAL_COORDINATES,
                    NUM_REWARDS_ON_MAP, NUM_TRAPS_ON_MAP,
                    reward_goal=REWARD_GOAL_VALUE,
                    reward_regular_step=REWARD_REGULAR_STEP_VALUE,
                    reward_trap=REWARD_TRAP_VALUE,
                    reward_collect=REWARD_COLLECT_VALUE,
                    initial_agent_points=INITIAL_AGENT_POINTS)

    # Der Agent wird nun mit der korrekten, erweiterten Anzahl von Zuständen initialisiert
    agent = QLearningAgent(num_states=env.num_states, # Verwendet die von Gridworld berechnete Anzahl
                           num_actions=env.num_actions,
                           learning_rate=LEARNING_RATE,
                           discount_factor=DISCOUNT_FACTOR,
                           epsilon=EPSILON_START,
                           epsilon_decay_rate=EPSILON_DECAY_RATE, 
                           min_epsilon=MIN_EPSILON)

    # --- Modus-Auswahl: Trainieren oder Laden ---      
    # Setze TRAIN_MODE auf True, um den Agenten zu trainieren und die Q-Tabelle zu speichern.
    # Setze TRAIN_MODE auf False, um eine bereits trainierte Q-Tabelle zu laden und zu evaluieren.
    TRAIN_MODE = False # Ändere dies zu False, um den Agenten zu laden und zu testen

    if TRAIN_MODE:
        print(f"Starte Training für {NUM_EPISODES} Episoden...")
        print(f"Gridworld Größe: {GRID_WIDTH}x{GRID_HEIGHT}, Start: {START_COORDINATES}, Ziele: {GOAL_COORDINATES}")
        print(f"Belohnungen: Ziel={REWARD_GOAL_VALUE}, Schritt={REWARD_REGULAR_STEP_VALUE}, Falle={REWARD_TRAP_VALUE}, Sammeln={REWARD_COLLECT_VALUE}")
        print(f"Startpunkte Agent: {INITIAL_AGENT_POINTS}")
        print(f"Q-Learning Parameter: Alpha={LEARNING_RATE}, Gamma={DISCOUNT_FACTOR}, Epsilon-Decay={EPSILON_DECAY_RATE:.8f}, Min-Epsilon={MIN_EPSILON}") # Formatierung für Epsilon-Decay
        print(f"Gesamtzahl der Zustände (erweitert): {env.num_states}") # NEU: Ausgabe der erweiterten Zustandsanzahl


        for episode in range(1, NUM_EPISODES + 1):
            current_state, current_agent_score = env.reset() 
            done = False
            episode_path = []
            score_history = []

            # Die Startposition für die Visualisierung muss aus dem erweiterten Zustand extrahiert werden
            start_x, start_y = env.get_coords_from_state(current_state)
            episode_path.append((start_x, start_y))
            score_history.append(current_agent_score)

            grid_at_episode_start = np.copy(env.grid) 

            for step_count in range(MAX_STEPS_PER_EPISODE):
                action = agent.choose_action(current_state)
                # env.step gibt jetzt den erweiterten nächsten Zustand zurück
                next_state, reward, done = env.step(current_state, action)
                agent.learn(current_state, action, reward, next_state, done)

                current_state = next_state
                current_agent_score += reward

                # Die aktuelle Position für die Visualisierung muss aus dem erweiterten Zustand extrahiert werden
                current_x, current_y = env.get_coords_from_state(current_state)
                episode_path.append((current_x, current_y))
                score_history.append(current_agent_score)

                if done:
                    break
            
            agent.decay_epsilon()

            if episode % 10000 == 0: # Ausgabe seltener für 1M Episoden
                print(f"Episode: {episode}/{NUM_EPISODES}, Endpunktzahl: {current_agent_score}, Schritte: {step_count+1}, Epsilon: {agent.epsilon:.6f}")
            
            if episode % VISUALIZE_EVERY_N_EPISODES == 0 or episode == NUM_EPISODES:
                print(f"Visualisiere Episode {episode} (Training)...")
                
                temp_visualizer = GridworldVisualizer(env, cell_size=60) 

                temp_visualizer.run_visualization(grid_data_at_episode_start=grid_at_episode_start, 
                                                  agent_pos_history=episode_path, 
                                                  score_history=score_history, 
                                                  fps=VISUALIZATION_FPS)
                
                print("Visualisierung beendet. Training wird fortgesetzt...")

        print("\nTraining abgeschlossen!")
        agent.save_q_table(Q_TABLE_FILENAME) # Q-Tabelle nach dem Training speichern

    else: # Wenn TRAIN_MODE auf False gesetzt ist
        print(f"Lade trainierten Agenten von '{Q_TABLE_FILENAME}'...")
        # Der Agent muss mit der korrekten Anzahl von Zuständen initialisiert werden, bevor geladen wird
        # Hier wird angenommen, dass die geladene Q-Tabelle die gleiche Zustandsgröße hat.
        # env.num_states wurde bereits in Gridworld.__init__ berechnet.
        if not agent.load_q_table(Q_TABLE_FILENAME):
            print("Konnte Q-Tabelle nicht laden. Bitte stellen Sie sicher, dass sie existiert und TRAIN_MODE zuvor auf True gesetzt wurde, um sie zu erstellen.")
            pygame.quit()
            exit() # Skript beenden, wenn Laden fehlschlägt

    # --- Evaluierungsphase (für trainierten oder geladenen Agenten) ---
    print("\nStarte Evaluierungsphase...")
    # Epsilon für die Evaluierung auf 0.0 setzen, da die adaptive Verhaltensweise aus der gelernten Policy kommen soll
    MIN_EPSILON_EVALUATION = 0.0 # Keine Exploration in der Evaluierung

    agent.epsilon = MIN_EPSILON_EVALUATION 
    print(f"Epsilon für Evaluierung auf {agent.epsilon:.4f} gesetzt.")
    
    eval_rewards = [] 

    for eval_episode in range(1, NUM_EVAL_EPISODES + 1):
        current_state, current_agent_score = env.reset() 
        done = False
        episode_path = []
        score_history = []

        start_x, start_y = env.get_coords_from_state(current_state)
        episode_path.append((start_x, start_y))
        score_history.append(current_agent_score)

        grid_at_episode_start = np.copy(env.grid) # Kopie der Karte für die Visualisierung

        for step_count in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(current_state) 

            next_state, reward, done = env.step(current_state, action)

            # Der Agent lernt in der Evaluierungsphase NICHT mehr
            # agent.learn(current_state, action, reward, next_state, done) # AUSKOMMENTIERT

            current_state = next_state
            current_agent_score += reward

            current_x, current_y = env.get_coords_from_state(current_state)
            episode_path.append((current_x, current_y))
            score_history.append(current_agent_score)

            if done:
                break
        
        eval_rewards.append(current_agent_score)
        print(f"Evaluierungs-Episode: {eval_episode}/{NUM_EVAL_EPISODES}, Endpunktzahl: {current_agent_score}, Schritte: {step_count+1}")

        # Visualisiere jede Evaluierungs-Episode
        print(f"Visualisiere Evaluierungs-Episode {eval_episode}...")
        temp_visualizer = GridworldVisualizer(env, cell_size=60)
        temp_visualizer.run_visualization(grid_data_at_episode_start=grid_at_episode_start, 
                                          agent_pos_history=episode_path, 
                                          score_history=score_history, 
                                          fps=VISUALIZATION_FPS)
        print("Visualisierung beendet. Evaluierung wird fortgesetzt...")

    print(f"\nEvaluierung abgeschlossen! Durchschnittliche Endpunktzahl über {NUM_EVAL_EPISODES} Episoden: {np.mean(eval_rewards):.2f}")

    # --- Q-Tabelle und Policy anzeigen (immer) ---
    print("\nDie vollständige Q-Tabelle des Agenten:")
    print(agent.q_table)

    print("\nGelernte Policy (beste Aktion pro Zustand):")
    policy = agent.get_policy(env)
    for i, action_name in enumerate(policy):
        # Die Koordinaten müssen aus dem erweiterten Zustand extrahiert werden
        x, y = env.get_coords_from_state(i)
        print(f"Zustand ({x},{y}): {action_name}")

    # Schließe Pygame am Ende des Skripts, falls es noch offen ist
    pygame.quit()
