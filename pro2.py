import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    accuracy_score
)
print("=" * 60)
print("   PROJECT 2 : DATA CLASSIFICATION USING AI")
print("   DecodeLabs | Batch 2026")
print("=" * 60)
iris = load_iris()
X = iris.data       
y = iris.target      
print("\n[STEP 1] DATASET LOADED")
print(f"  Dataset        : Iris Benchmark Dataset")
print(f"  Total Samples  : {X.shape[0]}  (Balanced - 50 per class)")
print(f"  Total Features : {X.shape[1]}")
print(f"  Feature Names  : {list(iris.feature_names)}")
print(f"  Classes (3)    : {list(iris.target_names)}")
print(f"\n  Sample Data (first 5 rows):")
print(f"  {'Sepal L':>8} {'Sepal W':>8} {'Petal L':>8} {'Petal W':>8} {'Class':>12}")
print(f"  {'-'*52}")
for i in range(5):
    print(f"  {X[i][0]:>8.1f} {X[i][1]:>8.1f} {X[i][2]:>8.1f} {X[i][3]:>8.1f} {iris.target_names[y[i]]:>12}")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n[STEP 2] FEATURE SCALING APPLIED")
print(f"  Method  : StandardScaler")
print(f"  Formula : z = (x - mean) / std")
print(f"  Result  : Mean=0, Variance=1 (Balanced features)")
print(f"  Reason  : KNN is distance-based — unscaled data causes bias")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size    = 0.2,
    random_state = 42,      
    shuffle      = True    
)
print(f"\n[STEP 3] TRAIN-TEST SPLIT DONE")
print(f"  Total Samples    : {len(X_scaled)}")
print(f"  Training Set     : {len(X_train)} samples  (80%)")
print(f"  Testing Set      : {len(X_test)}  samples  (20%)")
print(f"  Random State     : 42  (Fixed — reproducible results)")
print(f"  Shuffle          : True (Order bias removed)")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print(f"\n[STEP 4] KNN MODEL TRAINED")
print(f"  Algorithm    : K-Nearest Neighbors (KNN)")
print(f"  K Value      : 5  (Majority vote among 5 neighbors)")
print(f"  Principle    : Similar things exist in close proximity")
print(f"  Training on  : {len(X_train)} samples")
predictions = model.predict(X_test)
print(f"\n[STEP 5] PREDICTIONS MADE ON TEST DATA")
print(f"  Testing on   : {len(X_test)} unseen samples")
print(f"  Predicted    : {list(predictions)}")
print(f"  Actual       : {list(y_test)}")
correct   = sum(predictions == y_test)
incorrect = sum(predictions != y_test)
print(f"  Correct      : {correct} / {len(y_test)}")
print(f"  Incorrect    : {incorrect} / {len(y_test)}")
cm       = confusion_matrix(y_test, predictions)
f1       = f1_score(y_test, predictions, average='weighted')
accuracy = accuracy_score(y_test, predictions)
report   = classification_report(y_test, predictions, target_names=iris.target_names)
print(f"\n[STEP 6] MODEL EVALUATION")
print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  F1 Score     : {f1:.4f}  ({f1*100:.2f}%)")
print(f"\n  Confusion Matrix:")
print(f"              Predicted")
print(f"              Setosa  Versicolor  Virginica")
for i, row in enumerate(cm):
    print(f"  Actual {iris.target_names[i]:>10}  {row}")
print(f"\n  Classification Report:")
print(report)
error_rates = []
k_range     = range(1, 21)

for k in k_range:
    knn   = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - f1_score(y_test, preds, average='weighted'))
best_k = k_range[error_rates.index(min(error_rates))]
print(f"[STEP 7] OPTIMAL K VALUE")
print(f"  Best K Found : {best_k}  (Lowest error rate)")
fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')
fig.suptitle(
    'PROJECT 2 : DATA CLASSIFICATION USING AI\nDecodeLabs Industrial Training Kit | Batch 2026',
    fontsize=14, fontweight='bold', color='white', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            linewidths=2, linecolor='#0d1117',
            annot_kws={"size": 14, "weight": "bold"},
            ax=ax1)
ax1.set_title('Confusion Matrix', fontweight='bold',
              fontsize=11, color='white', pad=10)
ax1.set_ylabel('Actual', color='white', fontsize=9)
ax1.set_xlabel('Predicted', color='white', fontsize=9)
ax1.tick_params(colors='white')
ax1.set_facecolor('#161b22')
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(list(k_range), error_rates, 'o-',
         color='#58a6ff', markersize=7, linewidth=2)
ax2.axvline(x=5, color='#f85149', linestyle='--',
            linewidth=2, label='K=5 (Used)')
ax2.axvline(x=best_k, color='#3fb950', linestyle=':',
            linewidth=2, label=f'K={best_k} (Best)')
ax2.set_title('Elbow Method: Choosing K',
              fontweight='bold', fontsize=11, color='white', pad=10)
ax2.set_xlabel('K Value', color='white', fontsize=9)
ax2.set_ylabel('Error Rate', color='white', fontsize=9)
ax2.legend(fontsize=8, facecolor='#161b22', labelcolor='white')
ax2.tick_params(colors='white')
ax2.set_facecolor('#161b22')
ax2.grid(True, alpha=0.2)
ax3 = fig.add_subplot(gs[0, 2])
per_class_f1 = f1_score(y_test, predictions, average=None)
bars = ax3.bar(iris.target_names, per_class_f1,
               color=['#58a6ff', '#3fb950', '#f0883e'],
               edgecolor='white', linewidth=1.5)
ax3.set_ylim(0, 1.15)
ax3.set_title('F1 Score per Class',
              fontweight='bold', fontsize=11, color='white', pad=10)
ax3.set_ylabel('F1 Score', color='white', fontsize=9)
ax3.tick_params(colors='white')
ax3.set_facecolor('#161b22')
ax3.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, per_class_f1):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom',
             color='white', fontweight='bold', fontsize=11)
    ax4 = fig.add_subplot(gs[1, 0])
unique, counts = np.unique(y, return_counts=True)
ax4.bar(iris.target_names, counts,
        color=['#58a6ff', '#3fb950', '#f0883e'],
        edgecolor='white', linewidth=1.5)
ax4.set_title('Class Distribution\n(Balanced Dataset)',
              fontweight='bold', fontsize=11, color='white', pad=10)
ax4.set_ylabel('Number of Samples', color='white', fontsize=9)
ax4.tick_params(colors='white')
ax4.set_facecolor('#161b22')
ax4.grid(True, alpha=0.2, axis='y')
for i, v in enumerate(counts):
    ax4.text(i, v + 0.5, str(v), ha='center',
             color='white', fontweight='bold', fontsize=12)
ax5 = fig.add_subplot(gs[1, 1])
colors_map = {0: '#58a6ff', 1: '#3fb950', 2: '#f0883e'}
for cls in range(3):
    mask = y == cls
    ax5.scatter(X[mask, 2], X[mask, 3],
                c=colors_map[cls], label=iris.target_names[cls],
                alpha=0.7, s=50, edgecolors='white', linewidth=0.3)
ax5.set_title('Feature Space\n(Petal Length vs Petal Width)',
              fontweight='bold', fontsize=11, color='white', pad=10)
ax5.set_xlabel('Petal Length (cm)', color='white', fontsize=9)
ax5.set_ylabel('Petal Width (cm)', color='white', fontsize=9)
ax5.legend(fontsize=8, facecolor='#161b22', labelcolor='white')
ax5.tick_params(colors='white')
ax5.set_facecolor('#161b22')
ax5.grid(True, alpha=0.2)
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#161b22')
ax6.axis('off')
summary_text = (
    f"  MODEL SUMMARY\n"
    f"  {'─'*28}\n\n"
    f"  Dataset      :  Iris Benchmark\n"
    f"  Samples      :  150  (Balanced)\n"
    f"  Features     :  4\n"
    f"  Classes      :  3\n\n"
    f"  Algorithm    :  KNN (k=5)\n"
    f"  Scaling      :  StandardScaler\n"
    f"  Split        :  80% Train / 20% Test\n\n"
    f"  Accuracy     :  {accuracy*100:.2f}%\n"
    f"  F1 Score     :  {f1*100:.2f}%\n"
    f"  Correct      :  {correct} / {len(y_test)}\n"
    f"  Incorrect    :  {incorrect} / {len(y_test)}\n"
)
ax6.text(0.05, 0.95, summary_text,
         transform=ax6.transAxes,
         fontsize=9.5, verticalalignment='top',
         fontfamily='monospace', color='#e6edf3',
         bbox=dict(boxstyle='round,pad=0.8',
                   facecolor='#21262d',
                   edgecolor='#30363d',
                   linewidth=1.5))
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

plt.savefig('project2_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("\n[STEP 8] VISUALIZATION SAVED")
print("  File : project2_results.png")
print("\n" + "=" * 60)
print("  PROJECT 2 COMPLETE!")
print("  All requirements covered:")
print("  [✓] Dataset Loaded & Understood")
print("  [✓] Train-Test Split (80/20)")
print("  [✓] KNN Classification Applied")
print("  [✓] Confusion Matrix Generated")
print("  [✓] F1 Score Calculated")
print("=" * 60)



