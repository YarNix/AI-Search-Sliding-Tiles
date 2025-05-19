from typing import Callable, Generic, Generator, TypeVarTuple, Unpack
import customtkinter as ctk
from collections import deque
from heapq import heappush, heappop, nsmallest
import sys
from time import time as now
from random import choice, randint, random
import math

import pandas as pd

BLANK = ''
STEPS = ['U', 'D', 'L', 'R']
SOLVING: dict[str, dict[str, Callable]]
Ts = TypeVarTuple('Ts')

type StateData = tuple[int, int, int, int, int, int, int, int, int]

class PriorityItem(Generic[Unpack[Ts]]):
    def __init__(self, score: int, *rest: Unpack[Ts]):
        self.score = score
        self.data = rest
    def __lt__(self, other: 'PriorityItem[Unpack[Ts]]') -> bool:
        return self.score < other.score
    def __repr__(self) -> str:
        return (self.score, self.data).__repr__()

class State:
    def __init__(self, data: StateData, prev = None):
        self.data = data
        self.prev: State | None = prev
    def __repr__(self) -> str:
        return self.data.__repr__()

def neighbours(data: StateData) -> Generator[StateData, None, None]:
    idx = data.index(BLANK)
    x, y = idx // 3, idx % 3
    datalist: list[int] = list(data)
    for stepx, stepy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        if (0 <= x + stepx < 3 and 0 <= y + stepy < 3):
            move = (x + stepx) * 3 + (stepy + y)
            datalist[idx], datalist[move] = datalist[move], datalist[idx]
            yield (datalist[0], datalist[1], datalist[2], datalist[3], datalist[4], datalist[5], datalist[6], datalist[7], datalist[8])
            datalist[idx], datalist[move] = datalist[move], datalist[idx]

def get_path_fromto(end_state: State, path: list[StateData]):
    prev = end_state
    while prev is not None:
        path.append(prev.data)
        prev = prev.prev
    path.reverse()

def manhat_dist(start: StateData, goal: StateData) -> int:
    distance = 0
    for idx, value in enumerate(start):
        if value == BLANK:
            continue
        goal_idx = goal.index(value)
        distance += abs(idx % 3 - goal_idx % 3) + abs(idx // 3 - goal_idx // 3)
    return distance

def misplace(start: StateData, goal: StateData) -> int:
    score = 0
    for idx, value in enumerate(start):
        if value == BLANK:
            continue
        score += 1 if value != goal[idx] else 0
    return score

def bfs(start_data: StateData, end_data: StateData) -> list[StateData]:
    solution = []

    if start_data == end_data:
        return [start_data]
    
    state_queue: deque[State] = deque([State(start_data)])
    visited: set[StateData] = set([start_data])

    while len(state_queue) != 0:
        state = state_queue.popleft()
        for new_data in neighbours(state.data):
            if new_data == end_data:
                solution.append(new_data)
                get_path_fromto(state, solution)
                state_queue.clear()
                break
            new_state = State(new_data, state)
            if new_data in visited:
                continue
            visited.add(new_data)
            state_queue.append(new_state)    
    return solution

def dfs(start_data: StateData, end_data: StateData) -> list[StateData]:
    solution = []

    if start_data == end_data:
        return [start_data]
    
    state_stack: list[State] = [State(start_data)]
    visited: set[StateData] = set([start_data])

    while len(state_stack) != 0:
        state = state_stack.pop()
        for new_data in neighbours(state.data):
            if new_data == end_data:
                solution.append(new_data)
                get_path_fromto(state, solution)
                state_stack.clear()
                break
            new_state = State(new_data, state)
            if new_data in visited:
                continue
            visited.add(new_data)
            state_stack.append(new_state)
    return solution

def ids(start_data: StateData, end_data: StateData, max_depth = 31) -> list[StateData]:
    
    def dls(curr_state: StateData, end_data: StateData, depth: int, solution: list[StateData], visited_state: set[StateData]):
        if curr_state == end_data:
            return True
        if depth == 0:
            return False
        visited_state.add(curr_state)
        for new_data in neighbours(curr_state):
            if new_data not in visited_state:
                solution.append(new_data)
                if dls(new_data, end_data, depth - 1, solution, visited_state):
                    return True
                solution.pop()
        visited_state.remove(curr_state)
        return False

    solution = [start_data]
    depth = 0
    visited = set()
    while depth <= max_depth:
        visited.clear()
        if dls(start_data, end_data, depth, solution, visited):
            return solution
        depth += 1
    return []

def ucs(start_data: StateData, end_data: StateData) -> list[StateData]:
    solution = []

    if start_data == end_data:
        return [start_data]

    state_queue = [PriorityItem(0, State(start_data))]
    costs: dict = {}

    while len(state_queue):
        item = heappop(state_queue)
        curr_cost, state = item.score, item.data[0]
        for new_data in neighbours(state.data):
            new_cost = curr_cost + 1 
            if new_data == end_data:
                solution.append(new_data)
                get_path_fromto(state, solution)
                state_queue.clear()
                break
            if new_cost < costs.get(new_data, sys.maxsize):
                new_state = State(new_data, state)
                costs[new_data] = new_cost
                heappush(state_queue, PriorityItem(new_cost, new_state))
    return solution
    
def greedy(start_data: StateData, end_data: StateData) -> list[StateData]:
    solution = []
    
    next_state = State(start_data)
    visited: set[StateData] = set()

    while next_state is not None:
        current_state = next_state
        visited.add(current_state.data)
        next_state = None
        if current_state.data == end_data:
            get_path_fromto(current_state, solution)
            break
        heuristic = sys.maxsize
        for new_data in neighbours(current_state.data):
            temp = manhat_dist(new_data, end_data)
            if new_data in visited:
                continue
            if temp >= heuristic:
                continue
            heuristic = temp
            next_state = State(new_data, current_state)

    return solution

def Astar(start_data: StateData, end_data: StateData) -> list[StateData]:
    solution = []

    if start_data == end_data:
        return [start_data]

    state_queue = [PriorityItem(manhat_dist(start_data, end_data), State(start_data))]
    g_score: dict[StateData, int] = {start_data: 0}
    
    while len(state_queue) > 0:
        state, *_ = heappop(state_queue).data
        curr_g = g_score[state.data]
        for new_data in neighbours(state.data):
            new_g = curr_g + 1
            if new_data == end_data:
                solution.append(new_data)
                get_path_fromto(state, solution)
                state_queue.clear()
                break
            if new_g < g_score.get(new_data, sys.maxsize):
                new_state = State(new_data, state)
                g_score[new_data] = new_g
                new_h = manhat_dist(new_data, end_data)
                heappush(state_queue, PriorityItem(new_g + new_h, new_state))
    return solution

def IDAstar(start_data: StateData, end_data: StateData) -> list[StateData]:
    def dfs_limit(curr_state: StateData, g: int, threshold: int, solution: list[StateData], visited: set[StateData]) -> tuple[bool, int]:
        f = g + manhat_dist(curr_state, end_data)
        if f > threshold:
            return False, f
        if curr_state == end_data:
            return True, threshold
        
        min_cost = sys.maxsize
        visited.add(curr_state)
        for new_data in neighbours(curr_state):
            if new_data not in visited:
                solution.append(new_data)
                found, new_threshold = dfs_limit(new_data, g + 1, threshold, solution, visited)
                if found:
                    return True, threshold
                min_cost = min(min_cost, new_threshold)
                solution.pop()
        visited.remove(curr_state)
        return False, min_cost

    threshold = manhat_dist(start_data, end_data)
    solution = [start_data]
    
    while True:
        visited = set()
        found, new_threshold = dfs_limit(start_data, 0, threshold, solution, visited)
        if found:
            return solution
        if new_threshold == sys.maxsize:
            return []  # No solution found
        threshold = new_threshold

def hill_climb(start_data: StateData, end_data: StateData) -> list[StateData]:
    path = [start_data]
    curr_dist = manhat_dist(start_data, end_data)
    gen = neighbours(start_data)
    while True:
        next_state = next(gen, None)
        if next_state is None:
            return path
        next_dist = manhat_dist(next_state, end_data)
        if next_dist < curr_dist:
            gen = neighbours(next_state)
            curr_dist = next_dist
            path.append(next_state)

def hill_climb_steep(start_data: StateData, end_data: StateData) -> list[StateData]:
    path = [start_data]
    curr_data = start_data
    curr_dist = manhat_dist(start_data, end_data)
    while True:
        best_neighbour = None
        for next_state in neighbours(curr_data):
            dist = manhat_dist(next_state, end_data)
            if best_neighbour is None:
                if dist >= curr_dist:
                    continue
            elif dist >= best_neighbour[1]:
                continue
            best_neighbour = (next_state, dist)
        if best_neighbour is None:
            return path
        curr_data = best_neighbour[0]
        curr_dist = best_neighbour[1]
        path.append(curr_data)

def hill_climb_stoch(start_data: StateData, end_data: StateData) -> list[StateData]:
    path = [start_data]
    curr_dist = manhat_dist(start_data, end_data)
    neighs = [*neighbours(start_data)]
    while True:
        if len(neighs) == 0:
            return path
        next_state = neighs.pop(choice(range(len(neighs))))
        next_dist = manhat_dist(next_state, end_data)
        if next_dist < curr_dist:
            neighs = [*neighbours(next_state)]
            curr_dist = next_dist
            path.append(next_state)

def annealing(start_data: StateData, end_data: StateData) -> list[StateData]:
    temp = 100.0
    cooling = 0.99
    min_temp = 0.1

    current_data = start_data
    current_cost = manhat_dist(current_data, end_data)
    neighs = list(neighbours(current_data))

    path = [current_data]
    while temp > min_temp:
        if current_data == end_data:
            return path
        next_data = choice(neighs)
        next_cost = manhat_dist(next_data, end_data) 
        
        if next_cost < current_cost or random() < math.exp((current_cost - next_cost) / temp):
            current_data = next_data
            current_cost = next_cost
            neighs = list(neighbours(current_data))
            path.append(current_data)
        
        temp *= cooling  # Reduce the temperature
    
    return path

def beam_search(start_data: StateData, end_data: StateData, beam_width = 12) -> list[StateData]:
    def heu(data: StateData):
        return manhat_dist(data, end_data) * 100 + misplace(data, end_data) * 10 + int(random() * 10)

    beam = [ PriorityItem(heu(start_data), State(start_data)) ]

    while beam:
        candidates: list[PriorityItem[State]] = []
        for item in beam:
            curr_state, *_ = item.data
            for new_data in neighbours(curr_state.data):
                if new_data == end_data:
                    solution = [new_data]
                    get_path_fromto(curr_state, solution)
                    return solution
                if any((can.data[0].data == new_data for can in candidates)):
                    continue
                heappush(candidates, PriorityItem(heu(new_data), State(new_data, curr_state)))

        # Keep only top-k from the heap
        beam = nsmallest(beam_width, candidates)

    return []

Candidate = list[str]
def evolutionary(start_data: StateData, end_data: StateData) -> list[StateData]:
    def random_candidate(length: int) -> Candidate:
        return [choice(STEPS) for _ in range(length)]

    def follow_solution(start: StateData, solution: Candidate) -> StateData:
        curr = list(start)
        for step in solution:
            idx = curr.index(BLANK)
            x, y = divmod(idx, 3)
            dx, dy = 0, 0
            if step == "U": dx = -1
            elif step == "D": dx = 1
            elif step == "L": dy = -1
            elif step == "R": dy = 1
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_idx = nx * 3 + ny
                curr[idx], curr[new_idx] = curr[new_idx], curr[idx]
        return tuple(curr)

    def mutate(candidate: Candidate) -> Candidate:
        i = randint(0, len(candidate) - 1)
        candidate[i] = choice(STEPS)
        return candidate

    def crossover(parent1: Candidate, parent2: Candidate) -> Candidate:
        split = randint(1, len(parent1) - 2)
        return parent1[:split] + parent2[split:]

    # Parameters
    population_size = 300
    candidate_length = 30
    generations = 200
    mutation_rate = 0.1

    # Initial population
    population = [random_candidate(candidate_length) for _ in range(population_size)]
    for _ in range(generations):
        scored = [(candidate, manhat_dist(follow_solution(start_data, candidate), end_data)) for candidate in population]
        scored.sort(key=lambda x: x[1])
        # Early exit
        if scored[0][1] == 0:
            # Return the full trace from start to goal
            state_trace = [start_data]
            current = start_data
            for move in scored[0][0]:
                next_state = follow_solution(current, [move])
                if next_state == current:
                    continue
                state_trace.append(next_state)
                current = next_state
            return state_trace
        # Selection: top 20% survive
        survivors = [x[0] for x in scored[:population_size // 5]]
        # Generate new population via crossover and mutation
        new_population = survivors.copy()
        while len(new_population) < population_size:
            p1, p2 = choice(survivors), choice(survivors)
            child = crossover(p1, p2)
            if random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = new_population
    return []

def backtracking(grid: list[int], domains: dict[int, list[int]], is_valid: Callable[[list[int]], bool]) -> list[StateData]:
    if len(grid) == 9:
        if is_valid(grid):
            return [tuple(grid)]
        return []
    for var in (v for v in domains if v not in grid):
        for value in domains[var]:
            grid.append(value)
            result = backtracking(grid, domains, is_valid)
            if result:
                return result
            grid.pop()
    return []

def start_backtracking(start_data: StateData, end_data: StateData):
    end = [int(x) if x != BLANK else 0 for x in end_data]
    def is_valid(state: list[int]) -> bool:
        return state == end
    r = list(range(9))
    return backtracking(list(), domains={i: r for i in range(9)}, is_valid=is_valid)

def ac3(domains: dict[int, list[int]], constraint: Callable[[int, int, int, int], bool]) -> bool:
    variables = list(domains.keys())
    queue = deque((i, j) for i in variables for j in variables if i != j)

    def revise(xi: int, xj: int) -> bool:
        revised = False
        for x in domains[xi][:]:  # Copy of xi's domain
            if all(not constraint(xi, xj, x, y) for y in domains[xj]):
                domains[xi].remove(x)
                revised = True
        return revised

    while queue:
        xi, xj = queue.popleft()
        if revise(xi, xj):
            if not domains[xi]:
                return False
            for xk in variables:
                if xk != xi:
                    queue.append((xk, xi))
    return True

def start_ac3(start_data: StateData, end_data: StateData):
    domains = {i: list(range(9)) for i in range(9)}
    domains[0] = [1]  # Top-left must be 1
    domains[8] = [0]  # Bottom-right must be blank (0)
    def all_different_constraint(xi: int, xj: int, a: int, b: int) -> bool:
        return a != b
    if ac3(domains, all_different_constraint):
        return []
    end = [int(x) if x != BLANK else 0 for x in end_data]
    def is_valid(state: list[int]) -> bool:
        return state == end
    return backtracking(list(), domains, is_valid)



# Apply action to the board
def apply_action(state: StateData, action: str) -> StateData | None:
    idx = state.index(BLANK)
    x, y = divmod(idx, 3)
    dx, dy = 0, 0
    if action == 'U': dx = -1
    elif action == 'D': dx = 1
    elif action == 'L': dy = -1
    elif action == 'R': dy = 1

    nx, ny = x + dx, y + dy
    if not (0 <= nx < 3 and 0 <= ny < 3):
        return None

    new_idx = nx * 3 + ny
    state_list = list(state)
    state_list[idx], state_list[new_idx] = state_list[new_idx], state_list[idx]
    return tuple(state_list)

# Solver function using Q-table
def solve_q_learning(start_data: StateData, end_data: StateData) -> list[str]:
    MAX_STEPS = 100
    path = [start_data]
    current = start_data

    for _ in range(MAX_STEPS):
        if current == end_data:
            break
        state_str = str(tuple((int(x) if x != BLANK else 0 for x in current)))

        if state_str not in q_table.index:
            SOLVING['Reinforce Learning']['__message__'] = "Trạng thái không có trong Q-table."
            break

        # Pick action with max Q-value
        action = q_table.loc[state_str].idxmax()
        next_state = apply_action(current, action)

        if next_state is None or next_state == current:
            SOLVING['Reinforce Learning']['__message__'] = "Kẹt trong lòng lặp. Không có lời giải?"
            break

        path.append(next_state)
        current = next_state

    if current != end_data:
        return []
    elif '__message__' in SOLVING['Reinforce Learning']:
        del SOLVING['Reinforce Learning']['__message__']

    return path

class GridController:
    type MockData = tuple[str, str, str, str, str, str, str, str, str]
    def __init__(self, master, preset: MockData | None = None) -> None:
        self.selectedbtn = None
        if preset is None:
            preset = ('1', '2', '3', '4', '5', '6', '7', '8', BLANK)
        self.numberbtn = self.create_grid(master, preset)

    def create_grid(self, master, preset: MockData) -> list[ctk.CTkButton]:
        font = ctk.CTkFont(size=42)
        buttons: list[ctk.CTkButton] = []
        frame = ctk.CTkFrame(master, width=120 * 3, height=120 * 3)
        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                button = ctk.CTkButton(frame, text=preset[index], width=120, height=120, corner_radius=0, border_width=1, command=lambda i=index: self.on_btn_click(i), font=font)
                button.grid(row=i, column=j)
                buttons.append(button)
                if preset[index] == BLANK:
                    button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["border_color"])
        frame.grid(row=1, column=0, columnspan=5, padx=32)
        
        return buttons
    
    def on_btn_click(self, index: int):
        if self.selectedbtn is None:
            self.selectedbtn = self.numberbtn[index]
            self.selectedbtn.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        else:
            text, swap_text = self.selectedbtn.cget("text"), self.numberbtn[index].cget("text")
            self.numberbtn[index].configure(text=text)
            self.selectedbtn.configure(text=swap_text)
            if text == BLANK or swap_text == BLANK: 
                btn, empty_btn = (self.selectedbtn, self.numberbtn[index]) if text == BLANK else (self.numberbtn[index], self.selectedbtn)
                btn.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
                empty_btn.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["border_color"])
            else:
                self.selectedbtn.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            self.selectedbtn = None

class App:

    AVAILABLE_SOLVER: dict[str, dict[str, Callable]] = {
        'Uninformed Search': {
            'BFS': bfs,
            'DFS': dfs,
            'IDS': ids,
            'UCS': ucs
        },
        'Informed Search': {
            'Greedy': greedy,
            'A*': Astar,
            'IDA*': IDAstar
        },
        'Local Search': {
            'Hill Climb': hill_climb,
            'Steep Hill Climb': hill_climb_steep,
            'Stochastic Hill Climb': hill_climb_stoch,
        }, 
        'Better Local Search': {
            'Simulated Annealing': annealing,
            'Beam': beam_search,
            'Evolutionary': evolutionary
        },
        'Constraint satisfaction problems': {
            '__show_result__': False,
            'Backtracking search': start_backtracking,
            'AC-3': start_ac3
        },
        'Reinforce Learning': {
            'Q-Learning': solve_q_learning
        }
    }
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Sliding Puzzle")
        self.root.geometry("800x600")
        self.root.grid_anchor('center')
        title = ctk.CTkLabel(self.root, text="Bài toán 8 ô trượt", font=ctk.CTkFont(size=32))
        title.grid(row=0, column=0, columnspan=5, pady=3)
        self.grid = GridController(self.root)
        self.resultFrame = ctk.CTkFrame(self.root, fg_color='transparent')
        self.resultFrame.grid(row=4, column=0, pady=6, columnspan=5)
        self.result = ctk.CTkLabel(self.resultFrame, text="", font=ctk.CTkFont(size=16))
        self.result.grid(row=0, column=0, sticky="e")
        self.result_slider: ctk.CTkSlider | None = None
        self.end_state_data = ('1', '2', '3', '4', '5', '6', '7', '8', BLANK)
        solvebtn = ctk.CTkButton(self.root, text="Giải", width=120, height=40, command=self.solve)
        solvebtn.grid(row=5, column=2, padx=8)
        self.topic_selector = ctk.CTkComboBox(self.root, values=list(self.AVAILABLE_SOLVER.keys()), command=self.on_topic_changed)
        self.topic_selector.grid(row=5, column=0, columnspan=2, sticky="nw", pady=(0, 30))
        self.solver_selector = ctk.CTkComboBox(self.root, values=list(self.AVAILABLE_SOLVER[self.topic_selector.get()].keys()))
        self.solver_selector.grid(row=5, column=0, columnspan=2, sticky="sw", pady=(30, 0))
        editbtn = ctk.CTkButton(self.root, text="Sửa trạng thái\nmục tiêu", command=self.editor_open)
        editbtn.grid(row=5, column=3, columnspan=2, sticky="e")
        self.root.bind("<Left>", lambda e: self.leftright_update(-1))
        self.root.bind("<Right>", lambda e: self.leftright_update(1))
        self.solution: list[StateData] = []

    def leftright_update(self, offset):
        if self.result_slider is None:
            return
        new_value = int(self.result_slider.get()) + offset
        min, max = self.result_slider.cget("from_"), self.result_slider.cget("to")
        if new_value < min:
            new_value = min
        elif new_value > max:
            new_value = max
        self.result_slider.set(new_value)
        self.update_state_board(new_value)

    def update_state_board(self, index):
        for b, n in zip(self.grid.numberbtn, self.solution[int(index) - 1]):
            if str(n) == b.cget("text"):
                continue
            b.configure(text=n)
            if n == BLANK:
                b.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["border_color"])
            else:
                b.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

    def solve(self):
        
        if self.result_slider:
            self.result.configure(text="Đang giải...")
            self.result_slider.destroy()
            self.result_slider = None
            self.root.update()

        start_state_data = tuple(btn.cget("text") for btn in self.grid.numberbtn)
        t_begin = now()
        self.solution = self.AVAILABLE_SOLVER[self.topic_selector.get()][self.solver_selector.get()](start_state_data, self.end_state_data)
        t_end = now()
        
        if not self.AVAILABLE_SOLVER[self.topic_selector.get()].get('__show_result__', True):
            self.result.configure(text=f"Mất {t_end - t_begin:4f}s")
            return

        if len(self.solution) == 0:
            self.result.configure(text=f"Không tìm thấy\nMất {t_end - t_begin:4f}s")
        else:
            ln = len(self.solution)
            if self.solution[-1] == self.end_state_data:
                self.result.configure(text=f"Giải trong {ln - 1} bước\n{t_end - t_begin:4f}s")
            else:
                self.result.configure(text=f"Không giải được kẹt ở bước {ln - 1}\n{t_end - t_begin:4f}s")
            
            if ln > 1:
                self.result_slider = ctk.CTkSlider(self.resultFrame, from_=1, to=ln, number_of_steps=(ln - 1), command=self.update_state_board)
                self.result_slider.grid(row=0, column=1, pady=10, padx=4)
                self.result_slider.set(1)

    def editor_open(self):
        modal = ctk.CTkToplevel()
        modal.transient(root) 
        modal.title('Chỉnh sửa trạng thái')
        grid = GridController(modal, self.end_state_data)
        ctk.CTkLabel(modal, text='Trạng thái mục tiêu', font=ctk.CTkFont(size=32))\
        .grid(row=0, column=0, columnspan=5, pady=14)
        ctk.CTkButton(modal, text="Hủy", width=120, height=60, command=modal.destroy)\
        .grid(row=2, column=0, columnspan=2, pady=16, sticky='e')
        def on_ok():
            setattr(self, 'end_state_data', tuple(btn.cget("text") for btn in grid.numberbtn))
            modal.destroy()
        ctk.CTkButton(modal, text="Lưu", width=120, height=60, command=on_ok)\
        .grid(row=2, column=3, columnspan=2, pady=16, sticky='w')
        modal.grid_columnconfigure(0, minsize=60)
        modal.grid_columnconfigure(4, minsize=60)
        modal.grid_anchor('center')
        modal.wait_visibility()
        modal.grab_set()
        root.wait_window(modal)

    def on_topic_changed(self, new_topic: str):
        on_select = self.AVAILABLE_SOLVER[new_topic].get('__on_select__', None)
        if on_select:
            on_select(self)
        new_opts = [opt for opt in self.AVAILABLE_SOLVER[new_topic].keys() if not opt.startswith('__')]
        self.solver_selector.configure(values=new_opts)
        self.solver_selector.set(new_opts[0])

SOLVING = App.AVAILABLE_SOLVER
if __name__ == "__main__":
    q_table: pd.DataFrame = pd.read_pickle("qtable.pkl")
    root = ctk.CTk()
    app = App(root)
    root.mainloop()
