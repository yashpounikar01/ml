import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_entropy(data, label):
    prob = data[label].value_counts(normalize=True)
    return -np.sum(prob * np.log2(prob))

def calc_info_gain(feature, data, label):
    total_entropy = calc_entropy(data, label)
    feature_entropy = sum((data[feature] == val).mean() * calc_entropy(data[data[feature] == val], label) 
                          for val in data[feature].unique())
    return total_entropy - feature_entropy

def build_tree(data, label):
    if data[label].nunique() <= 1 or data.empty:
        return data[label].iloc[0] if not data.empty else None
    best_feature = max(data.columns.drop(label), key=lambda f: calc_info_gain(f, data, label))
    return {best_feature: {val: build_tree(data[data[best_feature] == val], label) for val in data[best_feature].unique()}}

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    return predict(tree[feature].get(instance[feature], None), instance)

def evaluate(tree, test_data, label):
    return sum(predict(tree, row) == row[label] for _, row in test_data.iterrows()) / test_data.shape[0]

def plot_tree(tree, parent_name, pos=None, level=0, width=2., vert_gap=0.4):
    if pos is None: 
        pos = {parent_name: (0, 0)}
    
    for i, child in enumerate(tree.keys()):
        child_name = f"{parent_name}_{child}"
        pos[child_name] = (pos[parent_name][0] + (i - (len(tree) - 1) / 2) * width, pos[parent_name][1] - vert_gap)
        
        # Draw edge
        plt.plot([pos[parent_name][0], pos[child_name][0]], [pos[parent_name][1], pos[child_name][1]], 'k-')
        
        # Draw node label
        if isinstance(tree[child], dict):
            plt.text(pos[child_name][0], pos[child_name][1], child, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'))
        else:
            plt.text(pos[child_name][0], pos[child_name][1], f'{child}\n{tree[child]}', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='black'))
        
        # Recursively plot the subtree
        if isinstance(tree[child], dict):
            plot_tree(tree[child], child_name, pos, level + 1, width / len(tree), vert_gap)

def draw_tree(tree):
    plt.figure(figsize=(10, 6))
    plot_tree(tree, 'root')
    plt.axis('off')
    plt.title('Decision Tree Structure', fontsize=16)
    plt.show()

if __name__ == "__main__":
    train_data = pd.read_csv("PlayTennis.csv").rename(columns=lambda x: x.strip())
    print("DataFrame:\n", train_data.head())
    print("Columns:", train_data.columns)
    label = 'Play'
    decision_tree = build_tree(train_data, label)
    print("Decision Tree Structure:\n", decision_tree)
    accuracy = evaluate(decision_tree, train_data, label)
    print(f"Accuracy on training data: {accuracy:.2f}")
    draw_tree(decision_tree)