import torch
from model import pde
import numpy as np
import time
from utils import read_csv, write_csv
from pathlib import Path


def train(
    train_loss,
    test_loss,
    poids,
    model,
    loss,
    optimizer,
    X_train,
    U_train,
    X_test_pde,
    X_test_data,
    U_test_data,
    X_pde,
    Re,
    f,
    x_std,
    y_std,
    u_mean,
    v_mean,
    p_std,
    t_std,
    u_std,
    v_std,
    folder_result,
    save_rate,
    batch_size,
    scheduler,
    X_border,
    X_border_test,
    time_simu, 
    mean_std
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()
    epoch = 0
    while (time.time()-time_start) < time_simu:
        epoch += 1
        loss_batch_train = {"total": [], "data": [], "pde": [], "border": []}
        for batch in range(len(X_pde) // batch_size):
            model.train()  # on dit qu'on va entrainer (on a le dropout)
            # loss du pde
            X_pde_batch = X_pde[batch *
                                batch_size: (batch + 1) * batch_size, :]
            pred_pde = model(X_pde_batch)
            pred_pde1, pred_pde2, pred_pde3 = pde(
                pred_pde,
                X_pde_batch,
                Re=Re,
                x_std=x_std,
                y_std=y_std,
                u_mean=u_mean,
                v_mean=v_mean,
                p_std=p_std,
                t_std=t_std,
                u_std=u_std,
                v_std=v_std,
            )
            loss_pde = (
                torch.mean(pred_pde1**2)
                + torch.mean(pred_pde2**2)
                + torch.mean(pred_pde3**2)
            )

            # loss des points de data
            pred_data = model(X_train)
            loss_data = loss(U_train, pred_data)

            # loss du border
            pred_border = model(X_border)
            goal_border = torch.tensor([-mean_std['u_mean']/mean_std['u_std'], -mean_std['v_mean']/mean_std['v_std']], dtype=torch.float32).expand(pred_border.shape[0], 2).to(device)
            loss_border_cylinder = loss(pred_border[:, :2], goal_border)  # (MSE)
            loss_totale = 1/3 * loss_data + 1/3 * loss_pde + 1/3 * loss_border_cylinder

            # Backpropagation
            loss_totale.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                loss_batch_train["total"].append(loss_totale.item())
                loss_batch_train["data"].append(loss_data.item())
                loss_batch_train["pde"].append(loss_pde.item())
                loss_batch_train["border"].append(loss_border_cylinder.item())

        # Pour le test :
        model.eval()

        # loss du pde
        test_pde = model(X_test_pde)
        test_pde1, test_pde2, test_pde3 = pde(
            test_pde,
            X_test_pde,
            Re=Re,
            x_std=x_std,
            y_std=y_std,
            u_mean=u_mean,
            v_mean=v_mean,
            p_std=p_std,
            t_std=t_std,
            u_std=u_std,
            v_std=v_std,
        )
        loss_test_pde = (
            torch.mean(test_pde1**2)
            + torch.mean(test_pde2**2)
            + torch.mean(test_pde3**2)
        )
        # loss de la data
        test_data = model(X_test_data)
        loss_test_data = loss(U_test_data, test_data)  # (MSE)

        # loss des bords 
        pred_border_test = model(X_border_test)
        goal_border_test = torch.tensor([-mean_std['u_mean']/mean_std['u_std'], -mean_std['v_mean']/mean_std['v_std']], dtype=torch.float32).expand(pred_border_test.shape[0], 2).to(device)
        loss_test_border = loss(pred_border_test[:, :2], goal_border_test)  # (MSE)

        # loss totale
        loss_test = 1/3 * loss_test_data + 1/3 * loss_test_pde + 1/3 * loss_test_border
        scheduler.step()
        with torch.no_grad():
            test_loss["total"].append(loss_test.item())
            test_loss["data"].append(loss_test_data.item())
            test_loss["pde"].append(loss_test_pde.item())
            test_loss["border"].append(loss_test_border.item())
            train_loss["total"].append(np.mean(loss_batch_train["total"]))
            train_loss["data"].append(np.mean(loss_batch_train["data"]))
            train_loss["pde"].append(np.mean(loss_batch_train["pde"]))
            train_loss["border"].append(np.mean(loss_batch_train["border"]))

        print(f"---------------------\nEpoch {epoch} :")
        print(f"---------------------\nEpoch {epoch} :", file=f)
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}"
        )
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}",
            file=f,
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}"
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}",
            file=f,
        )

        print(f"time: {time.time()-time_start:.0f}s")
        print(f"time: {time.time()-time_start:.0f}s", file=f)

        if (epoch) % save_rate == 0:
            dossier_midle = Path(
                folder_result + f"/epoch{len(train_loss['total'])}")
            dossier_midle.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                folder_result
                + f"/epoch{len(train_loss['total'])}"
                + "/model_weights.pth",
            )

            write_csv(
                train_loss,
                folder_result + f"/epoch{len(train_loss['total'])}",
                file_name="/train_loss.csv",
            )
            write_csv(
                test_loss,
                folder_result + f"/epoch{len(train_loss['total'])}",
                file_name="/test_loss.csv",
            )
    print('End training')
    return None