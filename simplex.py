#!/usr/bin/env python
"""simplex.py: Implementation of Simplex method."""

__author__= "Ruy Fuentes, Rodolfo Artavia, Esteban Aguilera"

import os.path
import numpy
import sys


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
    # 3,5 - (Objective Function Coefficient)
    # 2,1,<=,6 (Restrictions and signs coefficient)
    # -1,3,<=,9 - (Restrictions and signs coefficient)
    # 0,1,<=,4 - (Restrictions and signs coefficient)
    fileName = sys.argv[1]
    solutionFileName = os.path.splitext(fileName)[0] + "_solucion.txt"

    if not os.path.isfile(fileName):
        print("File does not exist.")
        return

    # @TODO: we are expecting the file to have a perfect format (?) - add error handling
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

    # TODO: this iteration is only for max and with all restricitions <= - we need to check if there is >= and = restrictions

    # This is a possible way to manage the content
    # For the matrix we only need the content, not the headers
    # the header and first col, we are going to use

    # 1. Clean the Objective Fn to be U - (variable decision 1) - (variable decision) 2 = 0
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

        if listProblemDescription[1] == 'max':
            startSimplexIterations(
                initialMatrix, numberDesicionVars, headers, rowsDescription, solutionFileName)
        elif listProblemDescription[1] == "min":
            # TODO
            print("min")
        else:
            print("invalid entered method")
    elif listProblemDescription[0] == '1':
        print("Gran M")
        if listProblemDescription[1] == 'max':
            print("max")
        elif listProblemDescription[1] == "min":
            print("min")
        else:
            print("invalid optimization method")
    elif listProblemDescription[0] == '2':
        print("2 fases")
        # TODO implement the min: min is a max * -1
        # This is happy path for the max

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
                newVars.append(f"x{indexVar+numberDesicionVars+1}")
                listCoefficientFnObj.append("0")
                editDescriptionList(listRestrictions, i)
            elif des == ">=":
                # restriction is => add artificial var and a excess var
                # the excess var is equal a -1
                newVars.append(f"s{indexVar+numberDesicionVars+1}")
                listCoefficientFnObj.append("0")
                editDescriptionList(listRestrictions, i, "-1")
                indexVar += 1
                newVars.append(f"a{indexVar+numberDesicionVars+1}")
                indexesWithAs.append(i)
                # add the artificial var to the objective as well? yes because you need to do the
                # operation of lines (renglones)
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
        # Now for each newVar with a you need to edit the objective function
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
        # remove the normal coefficients
        # TODO: check this doesnt save the reference
        # now you need to check if there are positive artificial vars and remove them
        # by subtracting the coefficients - lines of artificial variable
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
            matrix[row_index+1][col_index] = restriction[col_index]
    return matrix

#Replaces the values of each non basic variable
def editDescriptionList(listRestrictions, i, value="1"):
    for j in range(len(listRestrictions)):
        if j != i:
            listRestrictions[j].insert(
                len(listRestrictions[j])-2, "0")
        else:
            listRestrictions[j].insert(
                len(listRestrictions[j])-2, value)

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
            # then do JUST ONE iteration more to get the second solution - demonstrate there are multiple solutions
            # not required to show all multiple solutions
            # TODO: missing give the answer like:
            # x1 = n, x2 = n2, etc.. U = Total, ignoring the artificial variables
            break
        else:
            #   First get the less row[0][i] with i<VnBNumber - That is the PIVOT COLUMN
            cp_index = getIndexForLessN(matrix[0], len(matrix[0])-1)
            CP = matrix[:, cp_index]
            LD = matrix[:, len(matrix[0]) - 1]
            #   Then for each item in LD
            #   (starting with 1 - ignore the 0) (this is the matrix[0][matrix.len()-1]) do item/Pivot Column)
            #   Get the index of CP that is lesser of all the divisions (ignore the LD when value is 0)
            #   This index is the PIVOT ROW
            fp_index = getIndexLesserWhileDivByCP(LD, CP)

            if fp_index == -1:
                writeToFile("There is not possible answer because the problem is not bounded, U es no acotada",
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
                            currentRow[j] = round(currentRow[j] -
                                                  (abs(oldCP[i]) * FP[j]), 6)
                        else:
                            currentRow[j] = round(currentRow[j] +
                                                  (abs(oldCP[i]) * FP[j]), 6)

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

# Subtracts the lesser value from the LD column where the division is already been applied,
# So the algorithm can continue iterating in case it doesnt find
# and optimal solution and the solution is not "DEGENERADA"
def getIndexLesserWhileDivByCP(ld, cp):
    # TODO: can happen its a "SOLUCION DEGENERADA"
    # This happen when there is n results that are less and and equal
    # just notify the solution is "degenerada" but keep iterating
    resultIndex = -1
    for i in range(1, len(ld)):
        # omit 0 because is undefined
        if cp[i] > 0:
            if (resultIndex == -1) or (round(ld[i] / cp[i], 6) < round(ld[resultIndex]/cp[resultIndex], 6)):
                resultIndex = i
    return resultIndex

# Subtracts the lesser value from a row, this for the simplex method logic
# which requires to find the lesser value in order to locate de pivot column
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

#Method to edit the TXT file selected
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

#To Translate the logical matrix into a string, with the main purpose of showing on console
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
