import argparse, pickle, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    precision_score, recall_score, f1_score, roc_curve, auc, classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from k_lstm_model import reshape_sequences, append_kurtosis_feature, build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='pickle from preprocess.py')
    ap.add_argument('--model_out', default='results/k_lstm.h5')
    ap.add_argument('--plots_dir', default='results/figures')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()

    with open(args.data, 'rb') as f:
        D = pickle.load(f)
    Xtr, ytr = D['X_train'], D['y_train']
    Xte, yte = D['X_test'], D['y_test']
    L = D['seq_len']

    Xtr, ytr = reshape_sequences(Xtr, ytr, L)
    Xte, yte = reshape_sequences(Xte, yte, L)

    Xtr = append_kurtosis_feature(Xtr)
    Xte = append_kurtosis_feature(Xte)

    model = build_model(L, Xtr.shape[2])
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ck = ModelCheckpoint(args.model_out, save_best_only=True, monitor='val_loss')

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[es, ck], verbose=1
    )

    # Evaluate
    yprob = model.predict(Xte).ravel()
    yhat = (yprob >= 0.5).astype(int)

    print("MSE:", mean_squared_error(yte, yprob))
    print("MAE:", mean_absolute_error(yte, yprob))
    print("R2 :", r2_score(yte, yprob))
    print("Precision:", precision_score(yte, yhat))
    print("Recall   :", recall_score(yte, yhat))
    print("F1       :", f1_score(yte, yhat))
    print("\nClassification Report:\n", classification_report(yte, yhat))
    print("\nConfusion Matrix:\n", confusion_matrix(yte, yhat))

    # ROC
    fpr, tpr, _ = roc_curve(yte, yprob)
    roc_auc = auc(fpr, tpr)

    import os; os.makedirs(args.plots_dir, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend(loc='lower right')
    plt.savefig(f'{args.plots_dir}/roc.png', dpi=160)

    # Loss curve
    plt.figure()
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Training / Validation Loss')
    plt.savefig(f'{args.plots_dir}/loss.png', dpi=160)

if __name__ == '__main__':
    main()
