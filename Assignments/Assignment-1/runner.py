import platform
import subprocess

current_platform = platform.system()

if current_platform == "Darwin":
    print("\nRunning on MacOS Machine")
elif current_platform == "Windows":
    print("\nRunning on Windows Machine")
else:
    print("Unsupported platform")


def naive_bayes():
    print("\n - - - - - REPORT FOR NAIVE BAYES MODEL - - - - -\n")
    subprocess.call(["python", "naive_bayes.py"])


def linear_support_vector():
    print("\n - - - - - REPORT FOR LINEAR SUPPORT VECTOR MODEL - - - - -\n")
    subprocess.call(["python", "linear_support_vector.py"])


def linear_regression():
    print("\n - - - - - REPORT FOR LINEAR REGRESSION - - - - -\n")
    subprocess.call(["python", "linear_regression.py"])


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
      "Neural Networks & Deep Learning\n"
      "\t\tAssignment-1\n"
      "Author: Bhavesh Asanabada (700744873)\n"
      "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")


def main():
    while True:
        print("Available models to execute\n"
              "1. Naive Bayes Model (glass.csv)\n"
              "2. Linear Support Vector Model (glass.csv)\n"
              "3. Linear Regression Model (Salary_Data.csv)\n"
              "'Q' to QUIT")
        option = input("\nEnter the Model option: ")

        if option == '1':
            naive_bayes()
        elif option == '2':
            linear_support_vector()
        elif option == '3':
            linear_regression()
        elif option == 'Q' or 'q':
            break
        else:
            print("Incorrect selection")


if __name__ == "__main__":
    main()
