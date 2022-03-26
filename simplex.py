#!/usr/bin/env python
import os.path
import numpy
import sys
import cmath


class BASH_COLORS:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def main():
    # @TODO: implement the [-h] flag to run the help flow
    if len(sys.argv) <= 1:
        displayHelp()
        return
    if sys.argv[1] == "-h":
        displayHelp()
        return

    # Read the file, is going to be a format like:
    # 0,max,2,3 - (${method type [0=Simplex, 1=GranM, 2=DosFases]}, ${max or min}, ${decision variables}, ${restriccions #})
    # 3,5 - (coeficientes de la función objetivo)
    # 2,1,<=,6 (coeficientes de las restricciones y signo de restricción)
    # -1,3,<=,9 - (coeficientes de las restricciones y signo de restricción)
    # 0,1,<=,4 - (coeficientes de las restricciones y signo de restricción)
    fileName = sys.argv[1]
    solutionFileName = os.path.splitext(fileName)[0] + "_solucion.txt"

    if not os.path.isfile(fileName):
        print("File does not exist.")
        return

    # @TODO: we are expecting the file to have a perfect format (?) - add error handling
    # Start the parsing
    f = open(fileName, "r")
    lines = f.readlines()

    # lines[0] - problem info
    listProblemDescription = lines[0].strip().split(",")
    # lines[1] - coeficientes funcion objetivo
    listCoefficientFnObj = lines[1].strip().split(",")
    # lines[2:] - restricciones
    # For each line remove the new line char '\n'
    # result: ['2', '1', '<=', '6']
    listRestrictions = []
    for item in lines[2:]:
        listRestrictions.append(item.strip().split(","))

    # TODO: this iteration is only for max and with all restricitions <= - we need to check if there is >= and = restrictions

    # This is a possible way to manage the content
    # For the matrix we only need the content, not the headers
    # the header and first col, we are going to use

    # 1. Clean the function objetivo to be U - (variable decision 1) - (variable decision) 2 = 0
    # NOTE: basic variables = amount of restrictions
    # 2. Now we should turn all restrictions to be = (achive this by adding the basic variables)
    # TODO: Step 3 and 2 need to be implemented - for now we expect to have the restrictions in the form <=
    # 3. Create matrix
    # cols = number of restrictions + number of variables + 2 (VB & LD)
    # rows = 2 (VB & U) + len(basic variables) @NOTE: basic variables = # of restrictions for now
    numberDesicionVars = int(listProblemDescription[2])
    numberRestrictions = int(listProblemDescription[3])

    if listProblemDescription[0] == '0':
        # TODO: we can't use this method when restrictions are != <=
        # In this case all the restrictions are <=
        numberOfTotalVars = numberDesicionVars + numberRestrictions
        # cols = (amout of total vars) + LD
        # rows = (amout of restrictions or basic vars) + U
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

        if listProblemDescription[1] == 'max':
            startSimplexIterations(
                initialMatrix, numberDesicionVars, headers, rowsDescription, solutionFileName)
        elif listProblemDescription[1] == "min":
            # TODO
            print("min")
        else:
            print("invalid entered method")
    elif listProblemDescription[0] == '1':
        # TODO: Gran M
        # print(listRestrictions)

        vecTemp = []
        numberOfS = 0
        numberOfA = 0
        numberOfS2 = 0  # variable de apoyo

        # cuenta la cantida de variables nuevas
        for i in listRestrictions:
            positionSize = len(i) - 2
            if i[positionSize] == '<=':
                numberOfS += 1
            elif i[positionSize] == '=':
                numberOfA += 1
            elif i[positionSize] == '>=':
                numberOfS += 1
                numberOfA += 1

        # Agrega la cantida de variables nuevas a Z y las restricciones

        for i in range(numberDesicionVars+numberOfS+numberOfA + 1):
            vecTemp.append(complex(0, 0))
        for i in listRestrictions:
            for x in range(numberOfS+numberOfA):
                i.insert(-2, '0')

        # determina el signo de M
        if listProblemDescription[1] == 'max':
            Msing = -1
        elif listProblemDescription[1] == "min":
            Msing = 1
        else:
            print("invalid optimization method")

        # coloca los valores no imaginarios dentro de Z
        for x in range(numberDesicionVars):
            vecTemp[x] = int(listCoefficientFnObj[x]) * -1

        numberOfA = 0   # se reutiliza la variable

        # ciclo que asigna los valores de las nuevas variables a las restricciones
        for i in listRestrictions:
            positionSize = len(i) - 2

            if i[positionSize] == '<=':
                i[numberDesicionVars+numberOfS2] = '1'
                numberOfS2 += 1

            elif i[positionSize] == '=':
                i[numberDesicionVars + numberOfA+numberOfS] = '1'
                numberOfA += 1

                # agrega los valores imaginarios en Z
                for x in range(numberDesicionVars):
                    vecTemp[x] += complex(0, int(i[x]) * Msing)
                vecTemp[-1] += (complex(0, int(i[-1])))

            elif i[positionSize] == '>=':

                i[numberDesicionVars + numberOfS2] = '1'
                numberOfS2 += 1

                i[numberDesicionVars + numberOfA + numberOfS] = '1'
                numberOfA += 1

                # Agrega los numeros imaginarios y la S
                for x in range(numberDesicionVars):
                    vecTemp[x] += complex(0, int(i[x]) * Msing)
                vecTemp[-1] += (complex(0, int(i[-1])))
                vecTemp.insert(-1, complex(1, 1))
            else:
                print("wrong sign")

        # borra los = de las restricciones
        for i in listRestrictions:
            i.pop(-2)

        # transforma el el tipo de las restricciones
        for i in listRestrictions:
            for x in range(len(i)):
                i[x] = float(i[x])

        print(listRestrictions)
        listCoefficientFnObj = vecTemp
        print(listCoefficientFnObj)

        # ****************************** creacion de matriz
        Mheader = ["VB"]
        for i in range(numberDesicionVars):
            Mheader.append("X" + str(i + 1))
        for i in range(numberOfS):
            Mheader.append("S" + str(i+1))
        for i in range(numberOfA):
            Mheader.append("A" + str(i+1))
        Mheader.append("LD")
        print(Mheader)

        MrowsDescription = ["U"]
        for i in range(numberRestrictions):
            MrowsDescription.append("X" + str(i + 1+numberDesicionVars))
        print(MrowsDescription)

        matrixBigM = buildMatrixForBigM(listRestrictions, listCoefficientFnObj)
        # print(matrixBigM)

        startSimplexIterations(
            matrixBigM, numberDesicionVars, Mheader, MrowsDescription, solutionFileName)

        # ********************************

    elif listProblemDescription[0] == '2':
        print("2 fases")
        # TODO implement the min: min is a max * -1
        # This is happy path for the max

        # add the xn needed for each <=
        # add the yn (variable artificial) needed for each =
        # add the  -xn variable de exceso and the yn variable artificial needed for each >=
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
                newVars.append(f"x{indexVar+numberDesicionVars+1}")
                listCoefficientFnObj.append("0")
                editDescriptionList(listRestrictions, i)
            elif des == ">=":
                # restriction is => add artifical var and a excess var
                # the excess var is equal a -1
                newVars.append(f"s{indexVar+numberDesicionVars+1}")
                listCoefficientFnObj.append("0")
                editDescriptionList(listRestrictions, i, "-1")
                indexVar += 1
                newVars.append(f"a{indexVar+numberDesicionVars+1}")
                indexesWithAs.append(i)
                # add the articial var to the objetivo as well? yes because you need to do the
                # operation of reglones
                # TODO: when is min use a +1
                listCoefficientFnObj.append("-1")
                editDescriptionList(listRestrictions, i)
            else:
                # then is a '=' restriction is = add artifical var
                newVars.append(f"a{indexVar+numberDesicionVars+1}")
                indexesWithAs.append(i)
                # TODO: when is min use a +1
                listCoefficientFnObj.append("-1")
                editDescriptionList(listRestrictions, i)
            indexVar += 1
        # Now for each newVar with a you need to edit the function objetivo
        # when is a MAX function then use -Mx5..

        print(headers)
        print(newVars)
        print(listRestrictions)
        # now you can remove the des - the last item is always the LD
        for res in listRestrictions:
            del res[len(res) - 2]
            for i in range(len(res)):
                res[i] = float(res[i])
        for i in range(len(listCoefficientFnObj)):
            listCoefficientFnObj[i] = float(listCoefficientFnObj[i]) * -1
        print(listRestrictions)
        print("Fn")

        print(listCoefficientFnObj)
        print(indexesWithAs)
        # remove the coeficientes normales
        # TODO: check this doesnt save the reference
        # now you need to check if there are positive artificial vars and remove them
        # by substract the coeficientes - reglones que esa variable artificial
        newCoeficientes = listCoefficientFnObj
        for i in range(numberDesicionVars):
            newCoeficientes[i] = 0.0
        # LD in 0
        newCoeficientes.append(0.0)

        operationNewRowsFor0 = []
        operationNewRowsFor0.append(newCoeficientes)
        for index in indexesWithAs:
            operationNewRowsFor0.append(listRestrictions[index])
        print(operationNewRowsFor0)
        newRow = []

        for col_index in range(len(operationNewRowsFor0[0])):
            newResult = operationNewRowsFor0[0][col_index]
            for row_index in range(1, len(operationNewRowsFor0)):
                newResult -= operationNewRowsFor0[row_index][col_index]
            newRow.append(newResult)
        print(newRow)
        # now for each col do col[0] - col[n] - col[n]
        matrix = buildMatrixForTwoFases(listRestrictions, newRow)

        # now build the headers, and the RD
        # the row description would be all except for the type 's'
        headers = headers + newVars + ["LD"]
        RD = ["U"]

        for posible_rd in newVars:
            if "s" not in posible_rd:
                RD.append(posible_rd)

        print(matrix)
        print(headers)
        print(RD)
        startSimplexIterations(
            matrix, numberDesicionVars, headers, RD, solutionFileName)
        # [0.4,0.5]
        # [0.3,0.1,<=,2.7]
        # [0.5,0.5,=,6]
        # [0.6,0.4,>=,6]

    else:
        print("invalid entered method ")


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
            matrix[row_index+1][col_index] = restriction[col_index]
    return matrix


def editDescriptionList(listRestrictions, i, value="1"):
    for j in range(len(listRestrictions)):
        if j != i:
            listRestrictions[j].insert(
                len(listRestrictions[j])-2, "0")
        else:
            listRestrictions[j].insert(
                len(listRestrictions[j])-2, value)


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
            #   First get the less row[0][i] with i<VnBNumber - Thats the COLUMNA PIVOTE
            cp_index = getIndexForLessN(matrix[0], len(matrix[0])-1)
            CP = matrix[:, cp_index]
            LD = matrix[:, len(matrix[0]) - 1]
            #   Then for each item in LD (starting with 1 - ignore the 0) (this is the matrix[0][matrix.len()-1]) do item/Columna Pivote)
            #   Get the index of CP that is lesser of all the devisions (ignore the LD when value is 0)
            #   This index is the FILA PIVOTE
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

            # variable basica que sale lp_index
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


def formatFloatToPrint(num):
    return "{:.2f}".format(num)


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
    # use only the variables de desicion
    for i in range(1, len(rows_solution)):
        response += "," + formatFloatToPrint(rows_solution[i])
    return f'U = {formatFloatToPrint(LD[0])},({response})'


# ---------------------------------
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
            matrix[row_index+1][col_index] = restriction[col_index]
    return matrix
# --------------------------------


def getIndexLesserWhileDivByCP(ld, cp):
    # TODO: can happen its a SOLUCION DEGENERADA
    # This happen when there is n results that are less and and equal
    # just notify la solucion va a ser degenerada, pero seguir las iteraciones
    resultIndex = -1
    for i in range(1, len(ld)):
        # omit 0 because is undefined
        if cp[i] > 0:
            if (resultIndex == -1) or (round(ld[i] / cp[i], 6) < round(ld[resultIndex]/cp[resultIndex], 6)):
                resultIndex = i
    return resultIndex


def getIndexForLessN(row, end=-1):
    resultIndex = 0
    if end == -1:
        end = len(row)
    for i in range(1, end):
        if round(row[i], 6) < round(row[resultIndex], 6):
            resultIndex = i
    return resultIndex


def isRowPositive(row, end):
    for i in range(0, end):
        item = row[i]
        if item < 0:
            return False
    return True


def displayHelp():
    print('\n')
    print(f"{BASH_COLORS.WARNING}To run the program you need to pass the file with the correct input format as a parameter{BASH_COLORS.ENDC}")
    print("Example:")
    print('\n')
    print(f"{BASH_COLORS.OKGREEN}python simplex.py filename.txt{BASH_COLORS.ENDC}")
    print('\n')


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


def buildMatrix(coeficientesFn, coeficientesRestr, problemDescr):
    # cols = number of restrictions + number of variables + 2 (VB & LD)
    # rows = 2 (VB & U) + len(basic variables) @NOTE: basic variables = # of restrictions for now
    numberDesicionVars = int(problemDescr[2])
    numberRestrictions = int(problemDescr[3])
    numberOfTotalVars = numberDesicionVars + numberRestrictions
    # cols = (amout of total vars) + LD
    # rows = (amout of restrictions or basic vars) + U
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


def getPrintableMatrix(headers, rowsDescr, content):
    contentToPrint = ""
    tabChar = "\t"
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
