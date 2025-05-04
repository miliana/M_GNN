import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

seeds=[ 75285, 79513, 43054, 83341, 76475, 72028, 49358, 76003, 68822, 12663]
def generate_synthetic_data(
    num_patients=800,
    num_metabolites=107,
    num_diseases=231,
    num_pathways=2014,
    random_state=42
):
    np.random.seed(random_state)
    one_hot_features = ["Gender_Female", "Gender_Male", "Smoking_Past", "Smoking_Current"]
    numerical_features = ["Age", "BMI"]
    selected_metabolites = [f"Metabolite_{i}" for i in range(num_metabolites)]

    # Demographics
    female = np.random.binomial(1, 0.5, num_patients)
    male = 1 - female
    smoking_past = np.random.binomial(1, 0.2, num_patients)
    smoking_current = np.random.binomial(1, 0.3, num_patients)
    age = np.clip(np.random.normal(60, 10, num_patients), 18, 90)
    bmi = np.clip(np.random.normal(25, 4, num_patients), 15, 40)

    # Metabolites
    metabolites = np.random.randn(num_patients, num_metabolites)
    mask = np.random.rand(num_patients, num_metabolites) < 0.1
    metabolites[mask] = np.nan

    df = pd.DataFrame({
        "Gender_Female": female,
        "Gender_Male": male,
        "Smoking_Past": smoking_past,
        "Smoking_Current": smoking_current,
        "Age": age,
        "BMI": bmi
    })
    for i, col in enumerate(selected_metabolites):
        df[col] = metabolites[:, i]

    metabolite_sum = np.nansum(metabolites[:, :3], axis=1)
    cutoff = np.nanpercentile(metabolite_sum, 60)
    labels = ((metabolite_sum > cutoff) | (smoking_current == 1)).astype(int)
    y_label = pd.Series(labels, name="label")

    ranges = pd.DataFrame({
        "id": selected_metabolites,
        "normal_min": np.random.uniform(-1, 0, num_metabolites),
        "normal_max": np.random.uniform(1, 2, num_metabolites),
        "normal_mean": np.random.uniform(-0.5, 1.5, num_metabolites)
    })

    diseases = [f"Disease_{i}" for i in range(num_diseases)]
    dis_map = pd.DataFrame({
        "id": np.random.choice(selected_metabolites, num_metabolites),
        "Diseases": np.random.choice(diseases, num_metabolites)
    })

    pathways = [f"Pathway_{i}" for i in range(num_pathways)]
    path_map = pd.DataFrame({
        "id": np.random.choice(selected_metabolites, num_metabolites),
        "Pathways": np.random.choice(pathways + [None], num_metabolites)
    })

    return df, y_label, ranges, dis_map, path_map, one_hot_features, numerical_features, selected_metabolites


class ImprovedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.gat = GATConv(hidden_channels, hidden_channels, heads=4, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 4)
        self.conv2 = SAGEConv(hidden_channels * 4, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.bn4 = nn.BatchNorm1d(hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        self.edge_scale = nn.Parameter(torch.tensor(0.5))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        edge_weight = None
        if edge_attr is not None:
            edge_weight = torch.sigmoid(edge_attr[:, 0]) * self.edge_scale
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.bn2(self.gat(x, edge_index, edge_weight)))
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.bn3(self.conv2(x, edge_index)))
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.bn4(self.fc1(x)))
        x = F.dropout(x, self.dropout, self.training)
        return self.fc2(x)


def train_and_evaluate(data, model, optimizer, criterion, scheduler,
                       device, epochs=1500, patience=300):
    data = data.to(device)
    model = model.to(device)
    best_val_f1 = 0.0
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(data.x, data.edge_index, data.edge_attr)
            vals = F.softmax(out_val[data.val_mask], dim=1)[:, 1].cpu().numpy()
            val_t = data.y[data.val_mask].cpu().numpy()
            fpr, tpr, thr = roc_curve(val_t, vals)
            finite = np.isfinite(thr)
            thr, fpr, tpr = thr[finite], fpr[finite], tpr[finite]
            best_idx = np.argmax(tpr - fpr)
            best_threshold = thr[best_idx]
            preds_val = (vals >= best_threshold).astype(int)
            val_f1 = f1_score(val_t, preds_val)

        scheduler.step(loss.item())
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # Load best model and final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        out_all = model(data.x, data.edge_index, data.edge_attr)
        # Re-compute threshold on validation one final time
        vals = F.softmax(out_all[data.val_mask], dim=1)[:, 1].cpu().numpy()
        val_t = data.y[data.val_mask].cpu().numpy()
        fpr, tpr, thr = roc_curve(val_t, vals)
        thr = thr[np.isfinite(thr)]
        best_threshold = thr[np.argmax(tpr - fpr)]

        # Apply to test set
        test_vals = F.softmax(out_all[data.test_mask], dim=1)[:, 1].cpu().numpy()
        test_t = data.y[data.test_mask].cpu().numpy()
        preds_test = (test_vals >= best_threshold).astype(int)

        metrics = {
            'threshold': best_threshold,
            'test_acc': accuracy_score(test_t, preds_test),
            'test_precision': precision_score(test_t, preds_test),
            'test_recall': recall_score(test_t, preds_test),
            'test_f1': f1_score(test_t, preds_test),
            'test_auroc': roc_auc_score(test_t, test_vals)
        }
    return model, metrics


def build_graph(df, ranges, dis_map, path_map, one_hot, num_feats, mets):
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(idx, type='patient', label=int(row['label']),
                   dem=row[one_hot + num_feats].values.astype(np.float32),
                   met=row[mets].values.astype(np.float32))
    for _, r in ranges.iterrows():
        G.add_node(r['id'], type='metabolite', feats=r[['normal_min','normal_max','normal_mean']].values.astype(np.float32))
    for dis in dis_map['Diseases'].unique():
        G.add_node(dis, type='disease', feats=np.zeros(3, dtype=np.float32))
    for pw in path_map['Pathways'].dropna().unique():
        G.add_node(pw, type='pathway', feats=np.zeros(3, dtype=np.float32))
    for idx, row in df.iterrows():
        for m in mets:
            if not np.isnan(row[m]):
                G.add_edge(idx, m, weight=float(row[m]))
    for _, r in dis_map.iterrows():
        G.add_edge(r['id'], r['Diseases'], weight=1.0)
    for _, r in path_map.iterrows():
        if pd.notna(r['Pathways']):
            G.add_edge(r['id'], r['Pathways'], weight=1.0)
    return G


def extract_data(G):
    nodes = list(G.nodes())
    feats = []
    # Collect raw feature vectors and enforce float32
    for n in nodes:
        d = G.nodes[n]
        if d['type'] == 'patient':
            raw = np.hstack([d['dem'], d['met']])
        else:
            raw = d['feats']
        feats.append(np.asarray(raw, dtype=np.float32))

    # Pad all vectors to the same dimension
    max_dim = max(arr.shape[0] for arr in feats)
    padded = [np.pad(arr, (0, max_dim - arr.shape[0]), constant_values=0).astype(np.float32)
              for arr in feats]
    x = torch.from_numpy(np.vstack(padded)).float()

    # Build bidirectional edge index
    idx_map = {n: i for i, n in enumerate(nodes)}
    edge_list = []
    for u, v in G.edges():
        ui, vi = idx_map[u], idx_map[v]
        edge_list.append([ui, vi])
        edge_list.append([vi, ui])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Build labels and masks for patients
    y = torch.full((len(nodes),), -1, dtype=torch.long)
    patient_idxs = []
    for i, n in enumerate(nodes):
        if G.nodes[n]['type'] == 'patient':
            y[i] = G.nodes[n]['label']
            patient_idxs.append(i)

    tr, tmp = train_test_split(patient_idxs, test_size=0.3,
                                random_state=0, stratify=y[patient_idxs])
    val, te = train_test_split(tmp, test_size=0.5,
                               random_state=0, stratify=y[tmp])
    train_mask = torch.tensor([i in tr for i in range(len(nodes))], dtype=torch.bool)
    val_mask   = torch.tensor([i in val for i in range(len(nodes))], dtype=torch.bool)
    test_mask  = torch.tensor([i in te for i in range(len(nodes))], dtype=torch.bool)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )


def main():
    df, y_label, ranges, dis_map, path_map, one_hot, num_feats, mets = generate_synthetic_data()
    proc = df.copy()
    proc['label'] = y_label
    imp = KNNImputer(n_neighbors=2)
    proc[one_hot + num_feats + mets] = imp.fit_transform(proc[one_hot + num_feats + mets])
    sc1 = StandardScaler(); proc[num_feats] = sc1.fit_transform(proc[num_feats])
    sc2 = StandardScaler(); proc[mets] = sc2.fit_transform(proc[mets])
    sm = SMOTE(random_state=0)
    Xb, yb = sm.fit_resample(proc[one_hot + num_feats + mets], proc['label'])
    dfb = pd.DataFrame(Xb, columns=one_hot + num_feats + mets)
    dfb['label'] = yb

    G = build_graph(dfb, ranges, dis_map, path_map, one_hot, num_feats, mets)
    data = extract_data(G)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedGNN(in_channels=data.x.size(1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
    crit = nn.CrossEntropyLoss()
    _, results = train_and_evaluate(data, model, opt, crit, sch, device)
    print(results)


if __name__ == '__main__':
    main()
