import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def write_on_file(theta):
    with open('theta_values.txt', 'w') as file:
        file.write(f"{theta[0]}\n{theta[1]}")


def visulize_linear_regression(mileage, price, theta):
    plt.plot(mileage, price, 'rx')
    plt.plot(mileage, theta[0] * mileage + theta[1], 'b-')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.show()


def visualize_cost(iterations, cost_history):
    plt.plot(range(iterations), cost_history, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.title('Cost Function over Iterations')
    plt.show()


def denormalize_theta(theta, min_mileage, max_mileage, min_price, max_price):
    theta0 = theta[0] * (max_price - min_price) / (max_mileage - min_mileage)
    theta1 = (theta[1] * (max_price - min_price) + min_price
              - theta0 * min_mileage)
    return theta0, theta1


def train_model(new_mileage, new_price):
    alpha = 0.1
    iterations = 1000
    theta0 = 0
    theta1 = 0
    cost_history = []

    for i in range(iterations):
        temp1 = theta1 - alpha * np.sum([theta0 * new_mileage[j] + theta1 - new_price[j] for j in range(len(new_mileage))]) / len(new_mileage)
        temp0 = theta0 - alpha * np.sum([(theta0 * new_mileage[j] + theta1 - new_price[j]) * new_mileage[j] for j in range(len(new_mileage))]) / len(new_mileage)
        theta1 = temp1
        theta0 = temp0
        cost_function = np.sum([(theta0 * new_mileage[j] + theta1 - new_price[j]) ** 2 for j in range(len(new_mileage))]) / (2 * len(new_mileage))
        cost_history.append(cost_function)
    visualize_cost(iterations, cost_history)
    return theta0, theta1


def visualize_data(mileage, price):
    plt.plot(mileage, price, 'rx')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.show()


def main():
    try:
        dataset = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Model file not found")
    else:
        mileage = np.array(dataset['km'])
        price = np.array(dataset['price'])
        visualize_data(mileage, price)
        min_mileage = np.min(mileage)
        max_mileage = np.max(mileage)
        min_price = np.min(price)
        max_price = np.max(price)
        new_mileage = (mileage - min_mileage) / (max_mileage - min_mileage)
        new_price = (price - min_price) / (max_price - min_price)
        theta = train_model(new_mileage, new_price)
        theta = denormalize_theta(theta, min_mileage, max_mileage, min_price, max_price)
        visulize_linear_regression(mileage, price, theta)
        write_on_file(theta)


if __name__ == "__main__":
    main()
