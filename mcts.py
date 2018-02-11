import numpy as np

from mdp_tree import StateNode, StateActionNode

class MCTSSwp(object):
    """
    Monte Carlo Tree Search Planning Method
    Simple Progressive Widening
    For infinite action space finite state space
    """
    def __init__(exploration_parameter, alpha, default_policy_fn, model_fn):
        self.cp = exploration_parameter
        self.alpha = alpha
        self.default_policy_fn = default_policy_fn
        self.model_fn = model_fn

    def run(st, model, rollout_times):
        root_node = Node(st, parent=None, depth=0)

        # grow the tree for N times
        for t in range(rollout_times):
            self.grow_tree(root_node)

        action = self.best_action(root_node)

        return action

    def best_action(node):
        if node.num_children() == 0:
            print("no child in root node")
            action = default_policy_fn.get_action(node.state)
        else:
            qs = []
            acs = []
            for child in node.children:
                q = child.value()
                qs.append(q)
                acs.append(acs)
            qs = np.asarray(qs)
            best_q_idx = np.argmax(qs) 
            action = acs[best_q_idx]
        return best_action

    def select_action_spw(state_node):

        if (state_node.visited_times)**alpha > len(state_node.children):
            action = default_policy_fn.get_action(state_node.state)
            new_sa_node = StateActionNode(state_node.state, action, parent=state_node, depth=state_node.depth+1):
            state_node.add_child(new_sa_node)
        else:
            qs = []
            acs = []
            for child in node.children:
                q = child.value() + self.cp * np.sqrt((np.log(state_node.visited_times)/child.visited_times))
                qs.append(q)
                acs.append(acs)
            qs = np.asarray(qs)
            best_q_idx = np.argmax(qs) 
            action = acs[best_q_idx]

        return action

    def grow_tree(root_node):
        current_s_node = root_node
        cumulative_reward = 0.

        while True:
            if current_s_node.visited_times == 0 or node.num_children() == 0:
                break

            # select action add a (s,a) node into tree
            action = select_action_spw(current_s_node)
            new_sa_node = StateActionNode(current_s_node.state, action, parent=current_s_node, depth=current_s_node.depth+1):
            current_s_node.add_child(new_sa_node)

            # model generate next state add a (s) node into tree
            state_nxt, reward = model_fn(current_s_node.state, action)
            new_s_node = StateNode(state_nxt, parent=new_sa_node, depth=new_sa_node.depth+1):
            new_sa_node.add_child()

            current_s_node = new_s_node


        cumulative_reward = self.eval(current_s_node)