import heapq
import visualizer  as viz


def heuristic(a, b):
    (x1, y1), (x2, y2) = a, b
    return abs(x2 - x1) + abs(y2 - y1)


def a_star_search(graph, start, goal):
    """A* search
    
    Arguments:
        graph {WeightedGrid} -- [description]
        start {[type]} -- [description]
        goal {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    came_from[start] = None
    cost_so_far = {}
    cost_so_far[start] = 0
    cost_with_priority = {}
    cost_with_priority[start] = 0 # heuristic(start, goal)
    path = {}
    i = 0
    while len(frontier) > 0:
        (_, current) = heapq.heappop(frontier)
        path[current] = i
        i += 1
        if current == goal: break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + 1  # + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # including heuristic makes exploration towards goal a priority
                priority = new_cost + heuristic(next, goal)
                cost_with_priority[next] = priority
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current
    
    return came_from, cost_so_far, cost_with_priority, path


def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse() #  reverse (goal -> start) to (start -> goal)
    return path


def test():
    start = (2, 6)
    goal = (7, 8)
    width = 3
    came_from, cost_so_far, cost_with_priority, path = (
        a_star_search(viz.diagram4, start=start, goal=goal))

    # viz.draw_grid(viz.diagram4, width, point_to=came_from, start=start, goal=goal)
    # print()
    viz.draw_grid(viz.diagram4, width, number=cost_so_far, start=start, goal=goal)
    # viz.draw_grid(viz.diagram4, width, number=cost_with_priority, start=start, goal=goal)
    print()
    viz.draw_grid(viz.diagram4, width, 
        path=reconstruct_path(came_from, start=start, goal=goal))


test()