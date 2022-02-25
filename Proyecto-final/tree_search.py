from boardgame2 import ReversiEnv
import collections
import numpy as np

def get_valid_modes(state, env):
    valid_moves = env.get_valid((state, 1))
    valid_moves = np.argwhere(valid_moves == 1)
    if len(valid_moves) == 0:
        valid_moves = [env.PASS]
    return valid_moves

def bfs_cannonical(board_shape, verbose=0):
    #Para todos los estados posibles que se puedan tener se guarda todas las acciones posibles, cual seria el proximo estado segun cada una de esas acciones, que reward tendria al tomar cada accion, y si habria un ganador (en entorno deterministico)
    env = ReversiEnv(board_shape=board_shape)
    # Reinicio el entorno
    (board, first_player) = env.reset()
    # Guardo el estado formateado como vector
    state_tuple = tuple(first_player * board.reshape(-1))
    seen = set([])
    cannonical_states = {}
    # deque es como una pila pero con doble entrada (rear-front)
    queue = collections.deque([state_tuple])
    while queue:
        vertex = queue.popleft()
        #vuelvo a formatear el estado al cuadrado del tablero
        state = np.array(vertex).reshape(board_shape, board_shape)
        #obtengo las jugadas validas
        valid_moves = get_valid_modes(state, env)
        #controla que no haya ganador todavia
        if env.get_winner((state, 1)) is None:
            #Declara el diccionario para el estado
            cannonical_states[vertex] = {}
            #recorre todos los estados futuros posibles a partir de las acciones validas, de tal manera que explora todos los estados posibles hasta encontrar un ganador (estado final)
            # popleft hace que se tome siempre el estado de menor cantidad de jugadas en la cola (los de la izquierda) y los estados que corresponen a la siguiente jugada se almacenan al final de la cola
            for action in valid_moves:
                action = tuple(action)
                #Para cada accion del estado va a guardar si termino, quien gano y el proximo estado
                cannonical_states[vertex][action] = {}
                (next_state, _), reward, done, _ = env.next_step((state, 1), action)
                next_state = next_state * -1 # Cannonical Form
                node = tuple(next_state.reshape(-1))
                cannonical_states[vertex][action]['done'] = done
                cannonical_states[vertex][action]['winner'] = -1 * reward
                cannonical_states[vertex][action]['next_node'] = node
                #evita que se vuelva a mirar un estado por el cual ya se paso, se agrega a la cola un nuevo estado que nunca se visito
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        if verbose==1:
            print(f'{len(cannonical_states)}\r', end='')
        
    return cannonical_states