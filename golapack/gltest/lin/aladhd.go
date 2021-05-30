package lin

import "fmt"

// Aladhd prints header information for the driver routines test paths.
func Aladhd(path []byte) {
	var corz, sord bool
	var c1, c3, p2, sym string

	c1 = string(path[:1])
	c3 = string(path[2:3])
	p2 = string(path[1:3])
	sord = c1 == "S" || c1 == "D"
	corz = c1 == "C" || c1 == "Z"
	if !(sord || corz) {
		return
	}

	if p2 == "GE" {
		//        GE: General dense
		fmt.Printf("\n %3s drivers:  General dense matrices\n", path)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        7. Last n/2 columns zero\n    2. Upper triangular                8. Random, CNDNUM = sqrt(0.1/EPS)\n    3. Lower triangular                9. Random, CNDNUM = 0.1/EPS\n    4. Random, CNDNUM = 2             10. Scaled near underflow\n    5. First column zero              11. Scaled near overflow\n    6. Last column zero\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L * U - A )  / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 4)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf("   %2d: abs( WORK(1) - RPVGRW ) / ( max( WORK(1), RPVGRW ) * EPS )\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "GB" {
		//        GB: General band
		fmt.Printf("\n %3s drivers:  General band matrices\n", path)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Random, CNDNUM = 2              5. Random, CNDNUM = sqrt(0.1/EPS)\n    2. First column zero               6. Random, CNDNUM = 0.1/EPS\n    3. Last column zero                7. Scaled near underflow\n    4. Last n/2 columns zero           8. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L * U - A )  / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 4)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf("   %2d: abs( WORK(1) - RPVGRW ) / ( max( WORK(1), RPVGRW ) * EPS )\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "GT" {
		//        GT: General tridiagonal
		fmt.Printf("\n %3s drivers:  General tridiagonal\n", path)
		fmt.Printf(" Matrix types (1-6 have specified condition numbers):\n    1. Diagonal                        7. Random, unspecified CNDNUM\n    2. Random, CNDNUM = 2              8. First column zero\n    3. Random, CNDNUM = sqrt(0.1/EPS)  9. Last column zero\n    4. Random, CNDNUM = 0.1/EPS       10. Last n/2 columns zero\n    5. Scaled near underflow          11. Scaled near underflow\n    6. Scaled near overflow           12. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L * U - A )  / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 4)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Messages:\n")

	} else if p2 == "PO" || p2 == "PP" || p2 == "PS" {
		//        PO: Positive definite full
		//        PS: Positive definite full
		//        PP: Positive definite packed
		if sord {
			sym = "Symmetric"
		} else {
			sym = "Hermitian"
		}
		if c3 == "O" {
			fmt.Printf("\n %3s drivers:  %9s positive definite matrices\n", path, sym)
		} else {
			fmt.Printf("\n %3s drivers:  %9s positive definite packed matrices\n", path, sym)
		}
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = 0.1/EPS\n   *3. First row and column zero       8. Scaled near underflow\n   *4. Last row and column zero        9. Scaled near overflow\n   *5. Middle row and column zero\n   (* - tests error exits from %3sTRF, no test ratios are computed)\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U' * U - A ) / ( N * norm(A) * EPS ), or\n       norm( L * L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 4)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Messages:\n")

	} else if p2 == "PB" {
		//        PB: Positive definite band
		if sord {
			fmt.Printf("\n %3s drivers:  %9s positive definite band matrices\n", path, "Symmetric")
		} else {
			fmt.Printf("\n %3s drivers:  %9s positive definite band matrices\n", path, "Hermitian")
		}
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Random, CNDNUM = 2              5. Random, CNDNUM = sqrt(0.1/EPS)\n   *2. First row and column zero       6. Random, CNDNUM = 0.1/EPS\n   *3. Last row and column zero        7. Scaled near underflow\n   *4. Middle row and column zero      8. Scaled near overflow\n   (* - tests error exits from %3sTRF, no test ratios are computed)\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U' * U - A ) / ( N * norm(A) * EPS ), or\n       norm( L * L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 4)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Messages:\n")

	} else if p2 == "PT" {
		//        PT: Positive definite tridiagonal
		if sord {
			fmt.Printf("\n %3s drivers:  %9s positive definite tridiagonal\n", path, "Symmetric")
		} else {
			fmt.Printf("\n %3s drivers:  %9s positive definite tridiagonal\n", path, "Hermitian")
		}
		fmt.Printf(" Matrix types (1-6 have specified condition numbers):\n    1. Diagonal                        7. Random, unspecified CNDNUM\n    2. Random, CNDNUM = 2              8. First row and column zero\n    3. Random, CNDNUM = sqrt(0.1/EPS)  9. Last row and column zero\n    4. Random, CNDNUM = 0.1/EPS       10. Middle row and column zero\n    5. Scaled near underflow          11. Scaled near underflow\n    6. Scaled near overflow           12. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U'*D*U - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 4)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Messages:\n")

	} else if p2 == "SY" || p2 == "SP" {
		//        SY: Symmetric indefinite full
		//            with partial (Bunch-Kaufman) pivoting algorithm
		//        SP: Symmetric indefinite packed
		//            with partial (Bunch-Kaufman) pivoting algorithm
		if c3 == "Y" {
			fmt.Printf("\n %3s drivers:  %9s indefinite matrices, 'rook' (bounded Bunch-Kaufman) pivoting\n", path, "Symmetric")
		} else {
			fmt.Printf("\n %3s drivers:  %9s indefinite packed matrices, partial (Bunch-Kaufman) pivoting\n", path, "Symmetric")
		}
		fmt.Printf(" Matrix types:\n")
		if sord {
			fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")
		} else {
			fmt.Printf("    1. Diagonal                        7. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              8. Random, CNDNUM = 0.1/EPS\n    3. First row and column zero       9. Scaled near underflow\n    4. Last row and column zero       10. Scaled near overflow\n    5. Middle row and column zero     11. Block diagonal matrix\n    6. Last n/2 rows and columns zero\n")
		}
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Messages:\n")

	} else if p2 == "SR" || p2 == "SK" {
		//        SR: Symmetric indefinite full,
		//            with rook (bounded Bunch-Kaufman) pivoting algorithm
		//
		//        SK: Symmetric indefinite full,
		//            with rook (bounded Bunch-Kaufman) pivoting algorithm,
		//            ( new storage format for factors:
		//              L and diagonal of D is stored in A,
		//              subdiagonal of D is stored in E )
		fmt.Printf("\n %3s drivers:  %9s indefinite matrices, 'rook' (bounded Bunch-Kaufman) pivoting\n", path, "Symmetric")

		fmt.Printf(" Matrix types:\n")
		if sord {
			fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")
		} else {
			fmt.Printf("    1. Diagonal                        7. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              8. Random, CNDNUM = 0.1/EPS\n    3. First row and column zero       9. Scaled near underflow\n    4. Last row and column zero       10. Scaled near overflow\n    5. Middle row and column zero     11. Block diagonal matrix\n    6. Last n/2 rows and columns zero\n")
		}

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf(" Messages:\n")

	} else if p2 == "HA" {
		//        HA: Hermitian
		//            Aasen algorithm
		fmt.Printf("\n %3s drivers:  %9s indefinite matrices, 'Aasen' Algorithm\n", path, "Hermitian")

		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Messages:\n")

	} else if p2 == "HE" || p2 == "HP" {
		//        HE: Hermitian indefinite full
		//            with partial (Bunch-Kaufman) pivoting algorithm
		//        HP: Hermitian indefinite packed
		//            with partial (Bunch-Kaufman) pivoting algorithm
		if c3 == "E" {
			fmt.Printf("\n %3s drivers:  %9s indefinite matrices, 'rook' (bounded Bunch-Kaufman) pivoting\n", path, "Hermitian")
		} else {
			fmt.Printf("\n %3s drivers:  %9s indefinite packed matrices, partial (Bunch-Kaufman) pivoting\n", path, "Hermitian")
		}

		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Messages:\n")

	} else if p2 == "HR" || p2 == "HK" {
		//        HR: Hermitian indefinite full,
		//            with rook (bounded Bunch-Kaufman) pivoting algorithm
		//
		//        HK: Hermitian indefinite full,
		//            with rook (bounded Bunch-Kaufman) pivoting algorithm,
		//            ( new storage format for factors:
		//              L and diagonal of D is stored in A,
		//              subdiagonal of D is stored in E )
		fmt.Printf("\n %3s drivers:  %9s indefinite matrices, 'rook' (bounded Bunch-Kaufman) pivoting\n", path, "Hermitian")

		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf(" Messages:\n")

	} else {
		//        Print error message if no header is available.
		fmt.Printf("\n %3s:  No header available\n", path)
	}
	//     First line of header

	// Unused by f4go :  9891 FORMAT ( / 1 X , A3 , " drivers:  " , A9 , " indefinite packed matrices" , ", 'rook' (bounded Bunch-Kaufman) pivoting" )
	//
	//     GE matrix types
	//
	//
	//     GB matrix types
	//
	//
	//     GT matrix types
	//
	//
	//     PT matrix types
	//
	//
	//     PO, PP matrix types
	//
	//
	//     PB matrix types
	//
	//
	//     SSY, SSP, CHE, CHP matrix types
	//
	//
	//     CSY, CSP matrix types
	//
	//
	//     Test ratios
	//
	//
}
