# io_grupo8_proyecto1
Proyecto # 1 - Investigación de operaciones - TEC 2022

## Install/Setup

### Prerequisit
Python3
pip install numpy

### How to exc

```
python simplex.py [-h] [file name - use the complete path]
```

### Numpy Explanation

Numpy is the library used for matrix management mainly, consisting of multidimensional array objects and a collection of routines for processing those arrays. Using NumPy, mathematical and logical operations on arrays can be performed

<p>For this solution this was the main means for the matrix creation so its content can be managed, however does not support strings so common arrays were used for the headers of the matrix for example:<p>
<p>[VB, x1, x2, x3, LD] 	-	Common Array<p>
<p>[[0, 0, 0, 0, 2] -	Numpy<p>
<p>[1, 3, 1, 0, 2] -	Numpy<p>
<p>[2, 0, 0, 1, 2]] -	Numpy<p>

which is written on ordinary arrays, but the elements and values of the matrix are implemented on numpy 

### Cmath for Complex Numbers

This module provides access to mathematical functions for complex numbers. The functions in this module accept integers, floating-point numbers or complex numbers as arguments. They will also accept any Python object that has either a __complex__() or a __float__() method: these methods are used to convert the object to a complex or floating-point number, respectively, and the function is then applied to the result of the conversion.

```python
import cmath
``` 
## Problem Analysis 

<p>The acceptance criteria for the algorithm used in the solution is to get the parameters for the objective function in the tabular 
simplex method to be implemented, once this requirement is met, we proceed to the next phase which is the implementantion 
of the Grand M and Dual Phase methods which require simplex in order to find a final and optimal solution being the case, in short:<p>

 Grand M / Dual Phase --> Tabular Simplex


<p>For the Simplex Iterations there is the method called startSimplexIterations which receives the matrix to iterate, this matrix contains the Objective Function 
and Restrictions this matrix was created before hand with and auxiliar method in charge of creating the matrix depending of the method needed (just Simplex, Grand M, Dual Phase)<p>

### Matrix Building Methods 
```python
def buildMatrix(coeficientesFn, coeficientesRestr, problemDescr)
``` 

Is the function used for the creation of the first matrix with the data imported for the .txt file
<p>Txt File Structure:<p>
<ul>
<li>line[0] 	- Contains the Problem Info
<li>line[1]	- Contains the Objective Function Coefficients 
<li>line[2]	- Contains the Restrictions
</ul>

### Matrix Building Method for Big M method
```python
def buildMatrixForBigM(restrictions, fnsObjetivo):
cols = len(fnsObjetivo)
    rows = len(restrictions) + 1
    matrix = numpy.zeros([rows, cols]).astype(complex)
    for col_index in range(len(fnsObjetivo)):
        matrix[0][col_index] = fnsObjetivo[col_index]
    for row_index in range(len(restrictions)):
        restriction = restrictions[row_index]
        for col_index in range(len(restriction)):
            matrix[row_index+1][col_index] = restriction[col_index]
    return matrix
``` 
<p>Basically the method that returns matrix for the Big M Augmented Objective Function and Restrictions<p> 

### Matrix Building Method for Two Phases
```python
def buildMatrixForTwoFases(restrictions, fnsObjetivo):
    cols = len(fnsObjetivo)
    rows = len(restrictions) + 1
    matrix = numpy.zeros(
        [rows, cols])
    for col_index in range(len(fnsObjetivo)):
        matrix[0][col_index] = fnsObjetivo[col_index]
    for row_index in range(len(restrictions)):
        restriction = restrictions[row_index]
        for col_index in range(len(restriction)):
            matrix[row_index+1][col_index] = restriction[col_index]
    return matrix
``` 
<p>Method that returns matrix for the Two Phases Augmented Objective Function and Restrictions<p> 

### Pivot Element Seeking process 

This is the process where we get a matrix with all the values from the Objective Fn which would be represented by a U, W or Z, which is going to be the first row, and then the next rows will be the restrictions, 
to find the **pivot element** 
first we need to find the minimum value in the first row and that defines the **pivot column**, for example<p>
<ul>
<p>[VB, x1, x2, x3, LD] <p>
<p>[[0, -3, 0, 0, 0] The pivot column is be the one containing the -3<p> 
<p>[1, 1, 1, 0, 2] <p>
<p>[2, 2, 0, 1, 6]] <p>
<p> Next for the proccess of selecting the pivot row its required to divide the LD values by the element located in the Column element, For example:<p>
</ul>
<ul>
<p>[VB, x1, x2, x3, LD] <p>
<p>[[0, -3, 0, 0, 0] <p> 
<p>[1, 1, 1, 0, 2/1] <p>
<p>[2, 2, 0, 1, 6/2]] <p>
</ul>

### Pivot Indexes
```python
def getIndexLesserWhileDivByCP(ld, cp):
    resultIndex = -1
    for i in range(1, len(ld)):
        # omit 0 because is undefined
        if cp[i] > 0:
            if (resultIndex == -1) or (round(ld[i] / cp[i], 6) < round(ld[resultIndex]/cp[resultIndex], 6)):
                resultIndex = i
    return resultIndex
``` 
```python
def getIndexForLessN(row, end=-1):
    resultIndex = 0
    if end == -1:
        end = len(row)
    for i in range(1, end):
        if round(row[i], 6) < round(row[resultIndex], 6):
            resultIndex = i
    return resultIndex

``` 
<p>And the lesser value in this case is the 2/1 so thats the pivot row, for the pivot element finally being the 1 located in the index [1,1] of the matrix all this process is done by the methods above<p>

## Simplex Method

<p>The Simplex method is an approach to solving linear programming models by hand using slack variables, tableaus, and pivot variables as a 
means to finding the optimal solution of an optimization problem.  A linear program is a method of achieving the best outcome given a maximum 
or minimum equation with linear constraints.  Most linear programs can be solved using an online solver such as MatLab, but the Simplex method 
is a technique for solving linear programs by hand.  To solve a linear programming model using the Simplex method the following steps are necessary:<p>
<ul>
<li>	Standard form
<li>	Introducing slack variables
<li>	Creating the tableau
<li>	Pivot variables
<li>	Creating a new tableau
<li>	Checking for optimality
<li>	Identify optimal values
<ul/>

### Simplex Iterations in Code
```python
def startSimplexIterations(matrix, vnBNumber, H, RD, outputLocation):
    # Check if all the row[0][i] with i<VnBNumber  are >= 0
    estado = 0
    writeToFile(f'Estado: {estado}', outputLocation)
    str_matrix = getPrintableMatrix(
        H, RD, matrix)
    writeToFile(str_matrix, outputLocation)

    # TODO: missing the INITIAL ANSWER

    partial_answer = ""
    # Start the iterations
    while True:
        estado += 1

        if isRowPositive(matrix[0], len(matrix[0])-1):
            writeToFile(
                f"The final answer is {partial_answer}", outputLocation, BASH_COLORS.OKGREEN)
            # TODO: check if there is MULTIPLE SOLUTIONS - if yes then do one more iteration
            # for this check the vars in H that are not in rd are != 0, if there is one 0, if is the case
            # then do JUST ONE iteration more to get the second solution - es demostrar que hay soluciones multiples
            # no necesariamente mostrar todas esas posibles soluciones
            # TODO: missing give the answer like:
            # x1 = n, x2 = n2, etc.. U = Total, ignoring the variables artificiales
            break
        else:
            #   First get the less row[0][i] with i<VnBNumber - Thats the PIVOT COLUMN
            cp_index = getIndexForLessN(matrix[0], len(matrix[0])-1)
            CP = matrix[:, cp_index]
            LD = matrix[:, len(matrix[0]) - 1]
            #   Then for each item in LD (starting with 1 - ignore the 0) (this is the matrix[0][matrix.len()-1]) do item/Columna Pivote)
            #   Get the index of CP that is lesser of all the devisions (ignore the LD when value is 0)
            #   This index is the PIVOT ROW
            fp_index = getIndexLesserWhileDivByCP(LD, CP)

            if fp_index == -1:
                writeToFile("There is not possible answer because the problem is not bounded, U es no acotada",
                            outputLocation, BASH_COLORS.FAIL)
                break

            FP = matrix[fp_index]
            #   Interseccion entre COLUMNA PIVAOTE y FILA PIVOTE matrix[CP][FP] makes the numero pivote
            NP = matrix[fp_index][cp_index]
            entrante = H[cp_index + 1]
            saliente = RD[fp_index]
            response = f'VB entrante: {entrante}, VB saliente: {saliente}, Número Pivot: {NP}'

            # Basic Variable from lp_index
            # Now set the all CP to 0 except the fp_index, fp_index = 1
            # then operación de reglón: all the FP need to be / NP
            # CP already has the reference
            for i in range(len(FP)):
                FP[i] = round(FP[i] / NP, 6)
            # not calculate the rest of the rows
            # for all rows
            oldCP = numpy.array(CP, copy=True)
            for i in range(len(matrix)):
                # if not row pivote
                if i != fp_index:
                    currentRow = matrix[i]
                    for j in range(len(matrix[i])):
                        if oldCP[i] > 0:
                            currentRow[j] = round(currentRow[j] -
                                                  (abs(oldCP[i]) * FP[j]), 6)
                        else:
                            currentRow[j] = round(currentRow[j] +
                                                  (abs(oldCP[i]) * FP[j]), 6)

            # response
            writeToFile(response, outputLocation)
            writeToFile(f'Estado {estado}', outputLocation)
            # Remove the saliente and add the entrante
            RD[fp_index] = entrante
            str_matrix = getPrintableMatrix(
                H, RD, matrix)
            writeToFile(str_matrix, outputLocation)
            partial_answer = getPartialAnwser(matrix, H, RD)
            writeToFile(
                f'Respuesta Parcial: {partial_answer}', outputLocation)

``` 
###Concepts
<ul> 
<li>The big m method is a modified version of the simplex method in linear programming (LP) in which we assign a very large value (M) to each of the artificial variables

<li>In Two Phase Method, the whole procedure of solving a linear programming problem (LPP) involving artificial variables is divided into two phases.
In phase I, we form a new objective function by assigning zero to every original variable (including slack and surplus variables) and -1 to each of the artificial variables. Then we try to eliminate the artificial varibles from the basis. The solution at the end of phase I serves as a basic feasible solution for phase II. In phase II, the original objective function is introduced and the usual simplex algorithm is used to find an optimal solution.
</ul>

###Synthesis

<p>For the creation of this solution the first method implementend was the Tabular Simplex so it can be used on Big M and Two Phases, 
by accomplyshing an optimal code for the Simplex Solution it can be invoked wherever it is need so to proceed, next is the creation of Big M 
which consists mainly in building the matrix with the imaginary variables needed, once this matrix is built it would mean it already has the 
restrictions and OF for the Big M, so the next step is just to invoke simplex with that matrix. Same case for two phases  <p>