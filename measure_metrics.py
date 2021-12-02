import os
import matplotlib.pyplot as plt

# No walls, scared
nowalls_scared_a = []
nowalls_scared_b = []
# No walls, afraid
nowalls_afraid_a = []
nowalls_afraid_b = []
# No walls, confused
nowalls_confused_a = []
nowalls_confused_b = []

# Walls, scared
walls_scared_a = []
walls_scared_b = []
# Walls, afraid
walls_afraid_a = []
walls_afraid_b = []
# Walls, confused
walls_confused_a = []
walls_confused_b = []

measures_a = dict()
measures_b = dict()

layouts = ["large_filter", "large_filter_walls"]
ghosts = ["scared", "afraid", "confused"]

current_layout = None
current_ghost = None

for layout in layouts:
    current_layout = layout

    for ghost in ghosts:
        current_ghost = ghost

        measures_a[(layout, ghost)] = []
        measures_b[(layout, ghost)] = []
        for i in range(0, 100):
            print("%s %s : %d / 99" %(layout, ghost, i))
            os.system('python run.py --agentfile stopagent.py --bsagentfile bayesfilter.py --ghostagent %s '
                      '--nghosts 1 --layout %s --silentdisplay' % (ghost, layout))
            file = open("values.txt", "r")
            measures_a[(layout, ghost)].append(float(file.readline()))
            measures_b[(layout, ghost)].append(float(file.readline()))

    data_a = [measures_a[(layout, "scared")], measures_a[(layout, "afraid")], measures_a[(layout, "confused")]]
    data_b = [measures_b[(layout, "scared")], measures_b[(layout, "afraid")], measures_b[(layout, "confused")]]

    fig = plt.figure()
    plt.boxplot(data_a)
    plt.title("Layout %s" % layout)
    plt.xticks([1, 2, 3], ['Scared', 'Afraid', 'Confused'])
    plt.xlabel("Ghost Type")
    plt.ylabel("Belief measure (100 iterations with 100 steps)")
    fig.savefig('figures/%s_a.png' % layout)

    fig = plt.figure()
    plt.boxplot(data_b)
    plt.title("Layout %s" % layout)
    plt.xlabel("Ghost Type")
    plt.ylabel("Quality of Belief measure (100 iterations with 100 steps)")
    plt.xticks([1, 2, 3], ['Scared', 'Afraid', 'Confused'])
    fig.savefig('figures/%s_b.png' % layout)
