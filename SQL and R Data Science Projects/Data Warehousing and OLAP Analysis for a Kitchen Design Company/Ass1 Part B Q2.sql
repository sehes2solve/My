/*Part B Q2*/
/* Creating DW Database */
CREATE DATABASE "HSD_DW";

SELECT * FROM pg_database;

/* a) */
SELECT
    C.CustomerID,
    C.CustomerName,
    PS.Quantity,
    PS.Total
FROM
    CUSTOMER C
JOIN
    PRODUCT_SALES PS ON C.CustomerID = PS.CustomerID
JOIN
    TIMELINE T ON PS.TimeID = T.TimeID
WHERE
    T.Date >= DATE '2018-05-31' - INTERVAL '90 days'
    AND T.Date <= DATE '2018-05-31';

/* b) */
SELECT
    C.CustomerID,
    C.CustomerName,
    AVG(PS.Total) AS AverageOrder
FROM
    CUSTOMER C
JOIN
    PRODUCT_SALES PS ON C.CustomerID = PS.CustomerID
GROUP BY
    C.CustomerID
HAVING
    AVG(PS.Total) > (
        SELECT AVG(Total)
        FROM PRODUCT_SALES
    );

/* c) */

SELECT
    PS.CustomerID,
    C.CustomerName,
    PS.ProductNumber,
    P.ProductName,
    T.Date,
    LAG(T.Date) OVER (PARTITION BY PS.CustomerID ORDER BY T.Date) AS "End Date",
    T.Date - LAG(T.Date) OVER (PARTITION BY PS.CustomerID ORDER BY T.Date) AS Days_between_Product_Sales
FROM
    CUSTOMER C
JOIN
    PRODUCT_SALES PS ON C.CustomerID = PS.CustomerID
JOIN
    TIMELINE T ON PS.TimeID = T.TimeID
JOIN
    PRODUCT P ON PS.ProductNumber = P.ProductNumber
ORDER BY
    PS.CustomerID;

/* d) */

SELECT
    T.QuarterText,
    SUM(PS.Total) AS TotalSales
FROM
    PRODUCT_SALES PS
JOIN
    TIMELINE T ON PS.TimeID = T.TimeID
GROUP BY T.QuarterText;


