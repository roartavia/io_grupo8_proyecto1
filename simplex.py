#!/usr/bin/env python
"""simplex.py: Implementation of Simplex method."""
__author__ = "Ruy Fuentes, Rodolfo Artavia, Esteban Aguilera"

import sys
import cmath
import numpy
import os.path
from pickle import TRUE
from fileinput import filename


class BASH_COLORS:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def main():
    if len(sys.argv) <= 1:
        displayHelp()
        return
    if sys.argv[1] == "-h":
        displayHelp()
        return

    # Read the file, is going to be a format like:
    # 0,max,2,3 - (${method type [0=Simplex, 1=GranM, 2=DosFases]}, ${max or min}, ${decision variables}, ${restriccions #})
    # 3,5 - (Objective Function Coefficient)
    # 2,1,<=,6 (Restrictions and signs coefficient)
    # -1,3,<=,9 - (Restrictions and signs coefficient)
    # 0,1,<=,4 - (Restrictions and signs coefficient)
    fileName = sys.argv[1]
    solutionFileName = os.path.splitext(fileName)[0] + "_solucion.txt"

    if not os.path.isfile(fileName):
        print("File does not exist.")
        return

    # Start the parsing
    f = open(fileName, "r")
    lines = f.readlines()

    # lines[0] - Problem info
    listProblemDescription = lines[0].strip().split(",")
    # lines[1] - Objective Function Coefficient
    listCoefficientFnObj = lines[1].strip().split(",")
    # lines[2:] - Restrictions
    # For each line remove the new line char '\n'
    # result: ['2', '1', '<=', '6']
    listRestrictions = []
    for item in lines[2:]:
        listRestrictions.append(item.strip().split(","))

    # This is a possible way to manage the content
    # For the matrix we only need the content, not the headers
    # the header and first col, we are going to use
    # 1. Clean the Objective Fn to be U - (variable decision 1) - (variable decision) 2 = 0
    # NOTE: basic variables = amount of restrictions
    # 2. Now we should turn all restrictions to be = (achive this by adding the basic variables)
    # 3. Create matrix
    # cols = number of restrictions + number of variables + 2 (VB & LD)
    # rows = 2 (VB & U) + len(basic variables) @NOTE: basic variables = # of restrictions for now
    numberDesicionVars = int(listProblemDescription[2])
    numberRestrictions = int(listProblemDescription[3])

    if listProblemDescription[0] == '0':
        # In this case all the restrictions are <=
        numberOfTotalVars = numberDesicionVars + numberRestrictions
        # cols = (amount of total vars) + LD
        # rows = (amount of restrictions or basic vars) + U
        # 3. Create matrix
        initialMatrix = buildMatrix(
            listCoefficientFnObj, listRestrictions, listProblemDescription)
        # Numpy can' manage the matrix with the headers, headers and row description are manage in 2 separte arrays
        headers = ["VB"]
        for i in range(numberOfTotalVars):
            headers.append("x" + str(i + 1))
        headers.append("LD")
        # when finish add the LD
        rowsDescription = ["U"]
        i = numberDesicionVars + 1
        for j in range(numberRestrictions):
            rowsDescription.append("x" + str(i + j))

        if listProblemDescription[1] == "min":
            # Transform to max
            IS_MIN = True
            for col in initialMatrix:
                col[0] = col[0] * -1
            startSimplexIterations(
                initialMatrix, numberDesicionVars, headers, rowsDescription, solutionFileName, True)
        else:
            startSimplexIterations(
                initialMatrix, numberDesicionVars, headers, rowsDescription, solutionFileName)
    elif listProblemDescription[0] == '1':
        # TODO: Gran M
        vecTemp = []
        numberOfS = 0
        numberOfA = 0
        numberOfS2 = 0  # support variables
        MrowsDescription = ["U"]

        amountofS = 1  # places the variables in the left vector of the matrix
        amountofA = 1

        # count the number of new variables
        for i in listRestrictions:
            positionSize = len(i) - 2
            if i[positionSize] == '<=':
                numberOfS += 1
            elif i[positionSize] == '=':
                numberOfA += 1
            elif i[positionSize] == '>=':
                numberOfS += 1
                numberOfA += 1
            else:
                print("wrong sing")
                quit()

        # Add the number of new variables to Z and the constraints

        for i in range(numberDesicionVars + numberOfS + numberOfA + 1):
            vecTemp.append(complex(0, 0))
        for i in listRestrictions:
            for x in range(numberOfS + numberOfA):
                i.insert(-2, '0')

        # determine the sign of M
        if listProblemDescription[1] == 'max':
            Msing = -1
        elif listProblemDescription[1] == "min":
            Msing = 1
        else:
            print("invalid optimization method")
            quit()

        # put the non-imaginary values into Z
        for x in range(numberDesicionVars):
            vecTemp[x] = int(listCoefficientFnObj[x]) * -1

        numberOfA = 0  # variable is reused

        # loop that assigns the values of the new variables to the constraints
        for i in listRestrictions:
            positionSize = len(i) - 2

            if i[positionSize] == '<=':
                i[numberDesicionVars + numberOfS2] = '1'
                numberOfS2 += 1

                MrowsDescription.append("S" + str(amountofS))
                amountofS += 1

            elif i[positionSize] == '=':
                i[numberDesicionVars + numberOfA + numberOfS] = '1'
                numberOfA += 1

                MrowsDescription.append("A" + str(amountofA))
                amountofA += 1

                # add the imaginary values in the objective function
                for x in range(numberDesicionVars):
                    vecTemp[x] += complex(0, int(i[x]) * Msing)
                vecTemp[-1] += (complex(0, int(i[-1])))

            elif i[positionSize] == '>=':

                i[numberDesicionVars + numberOfS2] = '1'
                numberOfS2 += 1

                i[numberDesicionVars + numberOfA + numberOfS] = '1'
                numberOfA += 1

                MrowsDescription.append("A" + str(amountofA))
                amountofA += 1

                # Add the imaginary numbers and the S in the objective function
                for x in range(numberDesicionVars):
                    vecTemp[x] += complex(0, int(i[x]) * Msing)
                vecTemp[-1] += (complex(0, int(i[-1])))
                vecTemp.insert(-1, complex(1, 1))

        # delete the = from constraints
        for i in listRestrictions:
            i.pop(-2)

        # transforms the data type of the constraints
        for i in listRestrictions:
            for x in range(len(i)):
                i[x] = float(i[x])

        print(listRestrictions)
        listCoefficientFnObj = vecTemp
        print(listCoefficientFnObj)

        #  Matrix M creation
        Mheader = ["VB"]
        for i in range(numberDesicionVars):
            Mheader.append("X" + str(i + 1))
        for i in range(numberOfS):
            Mheader.append("S" + str(i + 1))
        for i in range(numberOfA):
            Mheader.append("A" + str(i + 1))
        Mheader.append("LD")
        print(Mheader)

        print(MrowsDescription)

        matrixBigM = buildMatrixForBigM(listRestrictions, listCoefficientFnObj)
        # print(matrixBigM)

        startSimplexIterationsWithM(
            matrixBigM, numberDesicionVars, Mheader, MrowsDescription, solutionFileName)

    elif listProblemDescription[0] == '2':
        # TODO implement the min: min is a max * -1
        IS_MIN = False
        if listProblemDescription[1] == "min":
            # TODO
            # do the conversion * -1 and save the flag to do the * -1 when give the answer
            IS_MIN = True
        # add the xn needed for each <=
        # add the yn (artificial variable) needed for each =
        # add the  -xn excess variable and the yn artificial variable needed for each >=
        headers = ["VB"]
        for i in range(numberDesicionVars):
            headers.append("x" + str(i + 1))

        newVars = []
        indexVar = 0
        indexesWithAs = []
        for i in range(len(listRestrictions)):
            restriction = listRestrictions[i]
            des = restriction[len(restriction) - 2]
            if des == "<=":
                # add basic var
                newVars.append(f"x{indexVar+numberDesicionVars + 1}")
                listCoefficientFnObj.append("0")
                editDescriptionList(listRestrictions, i)
            elif des == ">=":
                # restriction is => add artificial var and a excess var
                # the excess var is equal a -1
                newVars.append(f"s{indexVar+numberDesicionVars + 1}")
                listCoefficientFnObj.append("0")
                editDescriptionList(listRestrictions, i, "-1")
                indexVar += 1
                newVars.append(f"a{indexVar+numberDesicionVars + 1}")
                indexesWithAs.append(i)
                # add the artificial var to the objective as well? yes because you need to do the
                # operation of lines (renglones)
                # TODO: when is min use a +1
                if IS_MIN:
                    listCoefficientFnObj.append("1")
                else:
                    listCoefficientFnObj.append("-1")
                editDescriptionList(listRestrictions, i)
            else:
                # then is a '=' restriction is = add artifical var
                newVars.append(f"a{indexVar + numberDesicionVars + 1}")
                indexesWithAs.append(i)
                # TODO: when is min use a +1
                if IS_MIN:
                    listCoefficientFnObj.append("1")
                else:
                    listCoefficientFnObj.append("-1")
                # listCoefficientFnObj.append("-1")
                editDescriptionList(listRestrictions, i)
            indexVar += 1
        # Now for each newVar with a you need to edit the objective function
        # when is a MAX function then use -Mx5..

        # now you can remove the des - the last item is always the LD
        for res in listRestrictions:
            del res[len(res) - 2]
            for i in range(len(res)):
                res[i] = float(res[i])
        for i in range(len(listCoefficientFnObj)):
            listCoefficientFnObj[i] = float(listCoefficientFnObj[i]) * -1
            if IS_MIN:
                listCoefficientFnObj[i] = listCoefficientFnObj[i] * -1

        # remove the coeficient
        # now you need to check if there are positive artificial vars and remove them
        # by substract the coeficientes - reglones que esa variable artificial
        newCoeficientes = listCoefficientFnObj.copy()
        for i in range(numberDesicionVars):
            newCoeficientes[i] = 0.0
        # LD in 0
        newCoeficientes.append(0.0)

        operationNewRowsFor0 = []
        operationNewRowsFor0.append(newCoeficientes)
        for index in indexesWithAs:
            operationNewRowsFor0.append(listRestrictions[index])
        newRow = []

        for col_index in range(len(operationNewRowsFor0[0])):
            newResult = operationNewRowsFor0[0][col_index]
            for row_index in range(1, len(operationNewRowsFor0)):
                newResult -= operationNewRowsFor0[row_index][col_index]
            newRow.append(newResult)
        # now for each col do col[0] - col[n] - col[n]
        matrix = buildMatrixForTwoFases(listRestrictions, newRow)

        # now build the headers, and the RD
        # the row description would be all except for the type 's'
        headers = headers + newVars + ["LD"]
        RD = ["U"]

        for posible_rd in newVars:
            if "s" not in posible_rd:
                RD.append(posible_rd)

        writeToFile("FASE 1", solutionFileName)
        startSimplexIterations(
            matrix, numberDesicionVars, headers, RD, solutionFileName, IS_MIN)

        # now from the headers, remove the artificial vars
        # as well the matrix
        # from 1 to ignore the VB header
        i = 1
        while i < len(headers):
            if 'a' in headers[i]:
                # remove this i-1 from the matrix and the i from the headers
                del headers[i]
                # print(headers)
                matrix = numpy.delete(matrix, i-1, 1)
            else:
                i += 1

        # now set the fnObj from the beginning
        # if all the fnObj are positive you need to make an extra step to clean the positives
        # is the same step of reglones removing from col[0] - the reglón of that variable
        # the easier way to see it is
        # for each RD check the value in the U, and if is more than 0 then this RD need to be substract from U to make a new
        # U row without this possible value
        # FIRST add the fnOBJ for second fase
        for bv_index in range(numberDesicionVars):
            matrix[0][bv_index] = listCoefficientFnObj[bv_index]
        # what rows you need to proccess
        indexesWithMoreThan0InU = []
        NO_TIENE_SOLUCION_FACTIBLE = False
        for i in range(1, len(RD)):
            # check this RD[i] is positive in headers
            # get what is the index in the headers, this is i+1
            if RD[i] in headers:
                h_col_index = headers.index(RD[i]) - 1
                value_in_matrix = round(matrix[0][h_col_index], 6)
                if value_in_matrix > 0.0:
                    indexesWithMoreThan0InU.append([i, value_in_matrix])
            else:
                NO_TIENE_SOLUCION_FACTIBLE = True
                break
        if not NO_TIENE_SOLUCION_FACTIBLE:
            if len(indexesWithMoreThan0InU) > 0:
                newRow = []

                RU = matrix[0]
                for col_index in range(len(RU)):
                    for complex_2 in indexesWithMoreThan0InU:
                        row_index = complex_2[0]
                        static = complex_2[1]
                        RU[col_index] -= matrix[row_index][col_index] * static
            writeToFile("FASE 2", solutionFileName)
            startSimplexIterations(
                matrix, numberDesicionVars, headers, RD, solutionFileName, IS_MIN)
        else:
            writeToFile("El problema no tiene solución factible",
                        solutionFileName)
    else:
        print("Método invalido")
        displayHelp()

# Method that receives the array "restrictions" created from the txt file containing the restrictions in
# an array created with Numpy
# "fnsObjetivo" which is going to contain the new Objective Function needed for the Two Phases iteration
# This Method will return the matrix needed to proceed to the Simplex Process


def buildMatrixForTwoFases(restrictions, fnsObjetivo):
    cols = len(fnsObjetivo)
    # because the fnObjetivo
    rows = len(restrictions) + 1
    matrix = numpy.zeros(
        [rows, cols])
    # add fn headers
    for col_index in range(len(fnsObjetivo)):
        matrix[0][col_index] = fnsObjetivo[col_index]
    # add content
    for row_index in range(len(restrictions)):
        restriction = restrictions[row_index]
        for col_index in range(len(restriction)):
            matrix[row_index + 1][col_index] = restriction[col_index]
    return matrix

# Replaces the values of each non basic variable


def editDescriptionList(listRestrictions, i, value="1"):
    for j in range(len(listRestrictions)):
        if j != i:
            listRestrictions[j].insert(
                len(listRestrictions[j]) - 2, "0")
        else:
            listRestrictions[j].insert(
                len(listRestrictions[j]) - 2, value)


# This is the Core method which starts the Simplex iterations as its name suggests,
# receiving the Matrix, the Number of Decision Variables as "vnBNumber,
# H will be the headers which is a common array containing the String Values of the headers
# of the main matrix since numpy does not support chars and strings
# RD will contain a list like [U, x1, x2, x3] as the logical orientation where
# the Row Descriptions is stored, this array should be read like this:
# U
# x1
# x2
# x3
# And Finally OutputLocation which is the TextFile where the solution is written
def startSimplexIterations(matrix, vnBNumber, H, RD, outputLocation, isMin=False):
    # Check if all the row[0][i] with i<VnBNumber  are >= 0
    estado = 0
    writeToFile(f'Estado: {estado}', outputLocation)
    str_matrix = getPrintableMatrix(
        H, RD, matrix)
    writeToFile(str_matrix, outputLocation)

    partial_answer = ""
    # Start the iterations
    while True:
        estado += 1

        if isRowPositive(matrix[0], len(matrix[0]) - 1):
            writeToFile(
                f"Solución final es: {partial_answer}", outputLocation, BASH_COLORS.OKGREEN)
            # for this check the vars in H that are not in rd are != 0, if there is one 0, there are more solutions
            hasMultiples = False
            for i in range(vnBNumber + 1, len(H)-1):
                if H[i] not in RD:
                    if matrix[0][i-1] == 0:
                        hasMultiples = True
                        break
            if hasMultiples:
                writeToFile("Este problema tiene soluciones multiples",
                            outputLocation, BASH_COLORS.WARNING)

                writeToFile(getFinalAnswer(matrix, H, RD,
                            vnBNumber, isMin), outputLocation)
                break
        else:
            #   First get the less row[0][i] with i<VnBNumber - Thats the COLUMNA PIVOTE
            cp_index = getIndexForLessN(matrix[0], len(matrix[0]) - 1)
            CP = matrix[:, cp_index]
            LD = matrix[:, len(matrix[0]) - 1]
            #   Then for each item in LD (starting with 1 - ignore the 0) (this is the matrix[0][matrix.len()-1]) do item/Columna Pivote)
            #   Get the index of CP that is lesser of all the devisions (ignore the LD when value is 0)
            #   This index is the FILA PIVOTE
            fp_index = getIndexLesserWhileDivByCP(LD, CP, outputLocation)

            if fp_index == -1:
                writeToFile("No hay solución ya que U NO está ACOTADA",
                            outputLocation, BASH_COLORS.FAIL)
                break

            FP = matrix[fp_index]
            #   Intersection entre PIVOT COLUMN y PIVOT ROW matrix[CP][FP] makes the PIVOT ELEMENT
            NP = matrix[fp_index][cp_index]
            entrante = H[cp_index + 1]
            saliente = RD[fp_index]
            response = f'VB entrante: {entrante}, VB saliente: {saliente}, Número Pivot: {NP}'

            # basic variable that comes out lp_index
            # Now set the all CP to 0 except the fp_index, fp_index = 1
            # then row operation: all the FP need to be / NP
            # CP already has the reference
            for i in range(len(FP)):
                FP[i] = round(FP[i] / NP, 6)
            # not calculate the rest of the rows
            # for all rows
            oldCP = numpy.array(CP, copy=True)
            for i in range(len(matrix)):
                # if not Pivot Row
                if i != fp_index:
                    currentRow = matrix[i]
                    for j in range(len(matrix[i])):
                        if oldCP[i] > 0:
                            currentRow[j] = round(
                                currentRow[j] - (abs(oldCP[i]) * FP[j]), 6)
                        else:
                            currentRow[j] = round(
                                currentRow[j] + (abs(oldCP[i]) * FP[j]), 6)

            # response
            writeToFile(response, outputLocation)
            writeToFile(f'Estado {estado}', outputLocation)
            # Remove the out coming and add the incoming
            RD[fp_index] = entrante
            str_matrix = getPrintableMatrix(
                H, RD, matrix)
            writeToFile(str_matrix, outputLocation)
            partial_answer = getPartialAnwser(matrix, H, RD)
            writeToFile(
                f'Respuesta Parcial: {partial_answer}', outputLocation)


def startSimplexIterationsWithM(matrix, vnBNumber, H, RD, outputLocation):
    # Check if all the row[0][i] with i<VnBNumber  are >= 0
    matrix[0][len(matrix[0])-1] *= -1
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

        if isRowPositive(matrix[0], len(matrix[0]) - 1):
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
            #   First get the less row[0][i] with i<VnBNumber - Thats the COLUMNA PIVOTE
            cp_index = getIndexForLessN(matrix[0], len(matrix[0]) - 1)
            CP = matrix[:, cp_index]
            LD = matrix[:, len(matrix[0]) - 1]
            #   Then for each item in LD (starting with 1 - ignore the 0) (this is the matrix[0][matrix.len()-1]) do item/Columna Pivote)
            #   Get the index of CP that is lesser of all the devisions (ignore the LD when value is 0)
            #   This index is the FILA PIVOTE
            fp_index = getIndexLesserWhileDivByCP(LD, CP, outputLocation)

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

            # variable basica que sale lp_index
            # Now set the all CP to 0 except the fp_index, fp_index = 1
            # then operación de reglón: all the FP need to be / NP
            # CP already has the reference
            for i in range(len(FP)):
                if NP.imag == 0:
                    temp = complex(FP[i].real / NP.real, 0)
                else:
                    temp = complex(FP[i].real / NP.real, FP[i].imag / NP.imag)
                temp = complex(round(temp.real, 6), round(temp.imag, 6))
                FP[i] = temp

            # not calculate the rest of the rows
            # for all rows
            oldCP = numpy.array(CP, copy=True)
            for i in range(len(matrix)):
                # if not row pivote
                if i != fp_index:
                    # i es el pivote, j es el la fila, FP es fila pivote, de la segunda tabla y esta bien
                    currentRow = matrix[i]
                    # CurrentRow muestra cada celda de matriz
                    for j in range(len(matrix[i])):

                        temp = complex(
                            oldCP[i].real * FP[j].real, abs(oldCP[i].imag) * FP[j].imag)
                        temp = complex(round(temp.real, 3),
                                       round(temp.imag, 3))
                        # print('temporal')
                        # print(temp)
                        if oldCP[i].real > 0:  # oldCP es la columna de la fila pasada, esta bien
                            currentRow[j] = round(currentRow[j] - temp, 3)
                            #currentRow[j] = round(currentRow[j] -(abs(oldCP[i]) * FP[j]), 6)

                        else:
                            currentRow[j] = round(currentRow[j] + temp, 3)
                            #currentRow[j] = round(currentRow[j] +(abs(oldCP[i]) * FP[j]), 6)

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


def formatFloatToPrint(num):
    return "{:.2f}".format(num)

# Subtracts the values from the LD column to create the partial solution


def getPartialAnwser(matrix, h, rd):
    # 0 in the values that are in H but not in RD
    # LD/RD value when RD
    LD = matrix[:, len(matrix[0]) - 1]
    rows_solution = []
    # from 1 to ignore the VB header to -1 to ignore the LD
    for i in range(1, len(h) - 1):
        if h[i] in rd:
            # calculate
            pos_x = rd.index(h[i])
            value_ld = LD[pos_x]
            value_intersection = matrix[pos_x][i - 1]
            rows_solution.append(value_ld / value_intersection)
        else:
            rows_solution.append(0)

    response = formatFloatToPrint(rows_solution[0])
    # use only the decision variable
    for i in range(1, len(rows_solution)):
        response += "," + formatFloatToPrint(rows_solution[i])
    return f'U = {formatFloatToPrint(LD[0])},({response})'


def buildMatrixForBigM(restrictions, fnsObjetivo):
    cols = len(fnsObjetivo)
    # because the fnObjetivo
    rows = len(restrictions) + 1
    matrix = numpy.zeros([rows, cols]).astype(complex)
    # add fn headers
    for col_index in range(len(fnsObjetivo)):
        matrix[0][col_index] = fnsObjetivo[col_index]
    # add content
    for row_index in range(len(restrictions)):
        restriction = restrictions[row_index]
        for col_index in range(len(restriction)):
            matrix[row_index + 1][col_index] = restriction[col_index]
    return matrix


def getIndexLesserWhileDivByCP(ld, cp, filename):
    # TODO: can happen its a SOLUCION DEGENERADA
    # This happen when there is n results that are less and and equal
    # just notify la solucion va a ser degenerada, pero seguir las iteraciones
    resultIndex = -1
    for i in range(1, len(ld)):
        # omit 0 because is undefined
        if type(cp[i]) is numpy.complex128:
            if cp[i].imag != 0:
                if (resultIndex == -1) or (round(ld[i].imag / cp[i].imag, 6) < round(ld[resultIndex].imag / cp[resultIndex].imag, 6)):
                    resultIndex = i
                # TODO: missing if the M is the same
            else:
                if (resultIndex == -1) or (round(ld[i].real / cp[i].real, 6) < round(ld[resultIndex].real / cp[resultIndex].real, 6)):
                    resultIndex = i
        else:
            if cp[i] > 0:
                if (resultIndex == -1) or (round(ld[i] / cp[i], 6) < round(ld[resultIndex] / cp[resultIndex], 6)):
                    # check if is ==, if yes then is a SOLUCION DEGENERADA.
                    if resultIndex != -1 and (round(ld[i] / cp[i], 6) == round(ld[resultIndex]/cp[resultIndex], 6)):
                        writeToFile(
                            "Se encontraron dos valores posibles de variable saliente. Por lo tanto se dará una solución DEGENEDARA", filename)
                    resultIndex = i
    return resultIndex


def getFinalAnswer(matrix, h, rd, vNum, isMin):
    LD = matrix[:, len(matrix[0]) - 1]
    u = LD[0]
    if isMin:
        u = u * -1
    rows_solution = []
    # from 1 to ignore the VB header to -1 to ignore the LD
    for i in range(1, len(h) - 1):
        if h[i] in rd:
            # calculate
            pos_x = rd.index(h[i])
            value_ld = LD[pos_x]
            value_intersection = matrix[pos_x][i - 1]
            rows_solution.append(value_ld / value_intersection)
        else:
            rows_solution.append(0)
    # clean it
    rows_solution = rows_solution[:vNum]
    strSolution = ""
    for i in range(len(rows_solution)):
        strSolution += f"x{i + 1} = {formatFloatToPrint(rows_solution[i])} "
    return f'U = {formatFloatToPrint(u)} con {strSolution}'


# Subtracts the lesser value from a row, this for the simplex method logic
# which requires to find the lesser value in order to locate de pivot column
def getIndexForLessN(row, end=-1):
    resultIndex = 0

    if end == -1:
        end = len(row)

    for i in range(1, end):
        if type(row[i]) is numpy.complex128:
            if row[i].imag != 0:
                if round(row[i].imag, 6) < round(row[resultIndex].imag, 6):
                    resultIndex = i
            else:
                if round(row[i].real, 6) < round(row[resultIndex].real, 6):
                    resultIndex = i
        else:
            if round(row[i], 6) < round(row[resultIndex], 6):
                resultIndex = i
    return resultIndex


def isRowPositive(row, end):
    for i in range(0, end):
        item = row[i]
        if type(item) is numpy.complex128:
            if item.real < 0:
                return False
        else:
            if round(item, 3) < 0.0:
                return False
    return True


def displayHelp():
    print('\n')
    print(f"{BASH_COLORS.WARNING}Para ejecutar el programa, debe pasar el archivo con el formato de entrada correcto como parámetro {BASH_COLORS.ENDC}")
    print("Ejemplo:")
    print('\n')
    print(f"{BASH_COLORS.OKGREEN}python simplex.py filename.txt{BASH_COLORS.ENDC}")
    print('\n')

# Method to edit the TXT file selected


def writeToFile(content, fileName, color=""):
    if color == "":
        print(content)
    else:
        print(f"{color}{content}{BASH_COLORS.ENDC}")
    if os.path.exists(fileName):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    file = open(fileName, append_write)
    file.write(content + '\n')
    file.close()

# This method is meant to create the origin matrix from the problem instructions
# Creates the matrix straight from the txt file where the problem statements are


def buildMatrix(coeficientesFn, coeficientesRestr, problemDescr):
    # cols = number of restrictions + number of variables + 2 (VB & LD)
    # rows = 2 (VB & U) + len(basic variables) @NOTE: basic variables = # of restrictions for now
    numberDesicionVars = int(problemDescr[2])
    numberRestrictions = int(problemDescr[3])
    numberOfTotalVars = numberDesicionVars + numberRestrictions
    # cols = (amount of total vars) + LD
    # rows = (amount of restrictions or basic vars) + U
    matrix = numpy.zeros(
        [(numberRestrictions + 1), (numberOfTotalVars + 1)])
    currentRowIndex = 0
    # We can use the numberDesicionVars
    for coeficiente in coeficientesFn:
        # Move to left side of the expression
        matrix[0][currentRowIndex] = (float(coeficiente)) * -1
        currentRowIndex += 1

    diagonalStartIndex = numberDesicionVars
    for i in range(numberRestrictions):
        for j in range(numberDesicionVars):
            matrix[i + 1][j] = float(coeficientesRestr[i][j])
        # diagonal
        matrix[i + 1][diagonalStartIndex] = float(1)
        diagonalStartIndex += 1

    # Now set the LD
    lastRowIndex = numberOfTotalVars
    for i in range(numberRestrictions):
        matrix[i + 1][lastRowIndex] = coeficientesRestr[i][len(
            coeficientesRestr[i]) - 1]
    return matrix

# To Translate the logical matrix into a string, with the main purpose of showing on console


def getPrintableMatrix(headers, rowsDescr, content):
    contentToPrint = ""
    tabChar = "\t\t"
    enterChar = "\n"
    for header in headers:
        contentToPrint += header + tabChar
    contentToPrint += enterChar
    for i in range(len(rowsDescr)):
        rowInfo = rowsDescr[i]
        contentToPrint += rowInfo + tabChar
        currentContentRow = content[i]
        for contentItem in currentContentRow:
            contentToPrint += formatFloatToPrint(contentItem) + tabChar
        contentToPrint += enterChar
    return contentToPrint


# Run the project
main()
