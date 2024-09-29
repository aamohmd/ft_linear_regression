import sys


def predict_price(mileage) -> int:
    with open('theta_values.txt', 'r') as file:
        theta0, theta1 = [float(x) for x in file.read().split()]
    estimated_price = theta0 * mileage + theta1
    return int(estimated_price)


def main():
    try:
        if len(sys.argv) > 2:
            raise IndexError
        mileage = float(sys.argv[1])
        estimated_price = predict_price(mileage)
    except IndexError:
        print("Must provide one argument")
    except ValueError:
        print("Mileage should be a number")
    except FileNotFoundError:
        print("Model file not found")
    else:
        print("Estimated price is:", estimated_price)


if __name__ == "__main__":
    main()
