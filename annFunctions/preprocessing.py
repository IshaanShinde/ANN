def splitData(df):

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # extracting test set (20%) and storing the rest in a temp set
    xtemp, xtest, ytemp, ytest = train_test_split(x, y, test_size = 0.2, random_state = 44)

    # extracting the validation set (0.125 = 10% / 80% to get 10% of the original data)
    # and the remainder is the 70% train set
    xtrain, xval, ytrain, yval = train_test_split(xtemp, ytemp, test_size = 0.125, random_state = 44)

    return x, y, xtrain, xtest, xval, ytrain, ytest, yval

