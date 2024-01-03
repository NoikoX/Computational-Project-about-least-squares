from Functions import *
import matplotlib.pyplot as plt
import time

d = {
    1928: 37,
    1929: 46,
    1930: 40,
    1931: 42,
    1932: 42,
    1933: 55,
    1934: 39,
    1935: 32,
    1936: 55,
    1937: 57,
    1938: 77,
    1939: 59,
    1940: 50,
    1941: 59,
    1942: 48,
    1943: 40,
    1944: 40,
    1945: 57,
    1946: 61,
    1947: 69,
    1948: 65,
    1949: 66,
    1950: 67,
    1951: 83,
    1952: 92,
    1953: 82,
    1954: 74,
    1955: 75,
    1956: 67,
    1957: 70,
    1958: 69,
    1959: 96,
    1960: 86,
    1961: 62,
    1962: 81,
    1963: 45,
    1964: 74,
    1965: 59,
    1966: 51,
    1967: 58,
    1968: 48,
    1969: 40,
    1970: 40,
    1971: 41,
    1972: 50,
    1973: 51,
    1974: 61,
    1975: 43,
    1976: 45,
    1977: 60,
    1978: 49,
    1979: 69,
    1980: 42,
    1981: 66,
    1982: 75,
    1983: 60,
    1984: 62,
    1985: 61,
    1986: 63,
    1987: 63,
    1988: 54,
    1989: 44,
    1990: 83,
    1991: 57,
    1992: 50,
    1993: 53,
    1994: 49,
    1995: 33,
    1996: 54,
    1997: 48,
    1998: 54,
    1999: 70,
    2000: 80,
    2001: 65,
    2002: 63,
    2003: 63,
    2004: 73,
    2005: 80,
    2006: 78,
    2007: 76,
    2008: 105,
    2009: 98,
    2010: 95,
    2011: 114,
    2012: 115,
    2013: 100,
    2014: 110,
    2015: 112,
    2016: 116,
    2017: 99,
    2018: 90,
    2019: 86,
    2020: 85,
    2021: 68,
    2022: 70
}


def plot_regression(years, goals, a, b):
    years_range = np.arange(1900, 2101)

    predicted_goals = a * years_range + b

    plt.scatter(years, goals, color='blue', label='Original data')

    plt.plot(years_range, predicted_goals, color='red', label='Regression line')

    plt.xlabel('Year')
    plt.ylabel('Goals')
    plt.title('Goals vs Year')
    plt.legend()
    plt.show()


def plot_data(years, goals):
    plt.scatter(years, goals, color='blue', label='Real goals')

    plt.xlabel('Year')
    plt.ylabel('Goals')
    plt.title('Goals vs Year')
    plt.legend()
    plt.show()


def predict_goals(year, data):
    years = np.array(list(data.keys()))
    goals = np.array(list(data.values()))
    b = goals

    A = np.vstack([years, np.ones(len(years))]).T

    # least squares here
    # x = least_sqrs_with_inv(A, b)

    x = least_sqrs_with_QR(A, b)

    a, b = x

    predicted_goals = a * year + b

    return predicted_goals


year_to_predict = 2030
pre = time.perf_counter()

predicted_goals = predict_goals(year_to_predict, d)
post = time.perf_counter()

diff = post - pre

print(f"Predicted goals for {year_to_predict}: {predicted_goals:.2f}")
print("Time consumed : ",  diff)

years = np.array(list(d.keys()))
goals = np.array(list(d.values()))

A = np.vstack([years, np.ones(len(years))]).T
x = least_sqrs_with_QR(A, goals)
a, b = x

plot_regression(years, goals, a, b)
plot_data(years, goals)
