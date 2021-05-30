package lin

import "fmt"

// Alahd prints header information for the different test paths.
func Alahd(path []byte) {
	var corz, sord bool
	var c1, c3, eigcnm, p2, subnam, sym string

	c1 = string(path[:1])
	c3 = string(path[2:3])
	p2 = string(path[1:3])
	sord = c1 == "S" || c1 == "D"
	corz = c1 == "C" || c1 == "Z"
	if !(sord || corz) {
		return
	}
	//
	if p2 == "GE" {
		//        GE: General dense
		fmt.Printf("\n %3s:  General dense matrices\n", path)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        7. Last n/2 columns zero\n    2. Upper triangular                8. Random, CNDNUM = sqrt(0.1/EPS)\n    3. Lower triangular                9. Random, CNDNUM = 0.1/EPS\n    4. Random, CNDNUM = 2             10. Scaled near underflow\n    5. First column zero              11. Scaled near overflow\n    6. Last column zero\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L * U - A )  / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 5)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 6)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 7)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 8)
		fmt.Printf(" Messages:\n")

	} else if p2 == "GB" {
		//        GB: General band
		fmt.Printf("\n %3s:  General band matrices\n", path)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Random, CNDNUM = 2              5. Random, CNDNUM = sqrt(0.1/EPS)\n    2. First column zero               6. Random, CNDNUM = .01/EPS\n    3. Last column zero                7. Scaled near underflow\n    4. Last n/2 columns zero           8. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L * U - A )  / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 6)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "GT" {
		//        GT: General tridiagonal
		fmt.Printf("\n %3s:  General tridiagonal\n", path)
		fmt.Printf(" Matrix types (1-6 have specified condition numbers):\n    1. Diagonal                        7. Random, unspecified CNDNUM\n    2. Random, CNDNUM = 2              8. First column zero\n    3. Random, CNDNUM = sqrt(0.1/EPS)  9. Last column zero\n    4. Random, CNDNUM = 0.1/EPS       10. Last n/2 columns zero\n    5. Scaled near underflow          11. Scaled near underflow\n    6. Scaled near overflow           12. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L * U - A )  / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 6)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "PO" || p2 == "PP" {
		//        PO: Positive definite full
		//        PP: Positive definite packed
		if sord {
			sym = "Symmetric"
		} else {
			sym = "Hermitian"
		}
		if c3 == "O" {
			fmt.Printf("\n %3s:  %9s positive definite matrices\n", path, sym)
		} else {
			fmt.Printf("\n %3s:  %9s positive definite packed matrices\n", path, sym)
		}
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = 0.1/EPS\n   *3. First row and column zero       8. Scaled near underflow\n   *4. Last row and column zero        9. Scaled near overflow\n   *5. Middle row and column zero\n   (* - tests error exits from %3sTRF, no test ratios are computed)\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U' * U - A ) / ( N * norm(A) * EPS ), or\n       norm( L * L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 5)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 6)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 7)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 8)
		fmt.Printf(" Messages:\n")

	} else if p2 == "PS" {
		//        PS: Positive semi-definite full
		if sord {
			sym = "Symmetric"
		} else {
			sym = "Hermitian"
		}
		if c1 == "S" || c1 == "C" {
			eigcnm = "1E04"
		} else {
			eigcnm = "1D12"
		}
		fmt.Printf("\n %3s:  %9s positive definite packed matrices\n", path, sym)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal\n    2. Random, CNDNUM = 2              \n   *3. Nonzero eigenvalues of: D(1:RANK-1)=1 and D(RANK) = 1.0/%4s\n   *4. Nonzero eigenvalues of: D(1)=1 and  D(2:RANK) = 1.0/%4s\n   *5. Nonzero eigenvalues of: D(I) = %4s**(-(I-1)/(RANK-1))  I=1:RANK\n    6. Random, CNDNUM = sqrt(0.1/EPS)\n    7. Random, CNDNUM = 0.1/EPS\n    8. Scaled near underflow\n    9. Scaled near overflow\n   (* - Semi-definite tests )\n", eigcnm, eigcnm, eigcnm)
		fmt.Printf(" Difference:\n")
		fmt.Printf("   RANK minus computed rank, returned by %sPSTRF\n", c1)
		fmt.Printf(" Test ratio:\n")
		fmt.Printf("   norm( P * U' * U * P' - A ) / ( N * norm(A) * EPS ), or\n   norm( P * L * L' * P' - A ) / ( N * norm(A) * EPS )\n")
		fmt.Printf(" Messages:\n")

	} else if p2 == "PB" {
		//        PB: Positive definite band
		if sord {
			fmt.Printf("\n %3s:  %9s positive definite band matrices\n", path, "Symmetric")
		} else {
			fmt.Printf("\n %3s:  %9s positive definite band matrices\n", path, "Hermitian")
		}
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Random, CNDNUM = 2              5. Random, CNDNUM = sqrt(0.1/EPS)\n   *2. First row and column zero       6. Random, CNDNUM = 0.1/EPS\n   *3. Last row and column zero        7. Scaled near underflow\n   *4. Middle row and column zero      8. Scaled near overflow\n   (* - tests error exits from %3sTRF, no test ratios are computed)\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U' * U - A ) / ( N * norm(A) * EPS ), or\n       norm( L * L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 6)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "PT" {
		//        PT: Positive definite tridiagonal
		if sord {
			fmt.Printf("\n %3s:  %9s positive definite tridiagonal\n", path, "Symmetric")
		} else {
			fmt.Printf("\n %3s:  %9s positive definite tridiagonal\n", path, "Hermitian")
		}
		fmt.Printf(" Matrix types (1-6 have specified condition numbers):\n    1. Diagonal                        7. Random, unspecified CNDNUM\n    2. Random, CNDNUM = 2              8. First row and column zero\n    3. Random, CNDNUM = sqrt(0.1/EPS)  9. Last row and column zero\n    4. Random, CNDNUM = 0.1/EPS       10. Middle row and column zero\n    5. Scaled near underflow          11. Scaled near underflow\n    6. Scaled near overflow           12. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U'*D*U - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 6)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "SY" {
		//        SY: Symmetric indefinite full,
		//            with partial (Bunch-Kaufman) pivoting algorithm
		if c3 == "Y" {
			fmt.Printf("\n %3s:  %9s indefinite matrices, partial (Bunch-Kaufman) pivoting\n", path, "Symmetric")
		} else {
			fmt.Printf("\n %3s:  %9s indefinite packed matrices, partial (Bunch-Kaufman) pivoting\n", path, "Symmetric")
		}
		fmt.Printf(" Matrix types:\n")
		if sord {
			fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")
		} else {
			fmt.Printf("    1. Diagonal                        7. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              8. Random, CNDNUM = 0.1/EPS\n    3. First row and column zero       9. Scaled near underflow\n    4. Last row and column zero       10. Scaled near overflow\n    5. Middle row and column zero     11. Block diagonal matrix\n    6. Last n/2 rows and columns zero\n")
		}
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 5)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 6)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 7)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 8)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 9)
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
		fmt.Printf("\n %3s:  %9s indefinite matrices, 'rook' (bounded Bunch-Kaufman) pivoting\n", path, "Symmetric")

		fmt.Printf(" Matrix types:\n")
		if sord {
			fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")
		} else {
			fmt.Printf("    1. Diagonal                        7. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              8. Random, CNDNUM = 0.1/EPS\n    3. First row and column zero       9. Scaled near underflow\n    4. Last row and column zero       10. Scaled near overflow\n    5. Middle row and column zero     11. Block diagonal matrix\n    6. Last n/2 rows and columns zero\n")
		}

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: ABS( Largest element in L )\n             - ( 1 / ( 1 - ALPHA ) ) + THRESH\n", 3)
		fmt.Printf("       where ALPHA = ( 1 + SQRT( 17 ) ) / 8\n")
		fmt.Printf("   %2d: Largest 2-Norm of 2-by-2 pivots\n             - ( ( 1 + ALPHA ) / ( 1 - ALPHA ) ) + THRESH\n", 4)
		fmt.Printf("       where ALPHA = ( 1 + SQRT( 17 ) ) / 8\n")
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 6)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "SP" {
		//        SP: Symmetric indefinite packed,
		//            with partial (Bunch-Kaufman) pivoting algorithm
		if c3 == "Y" {
			fmt.Printf("\n %3s:  %9s indefinite matrices, partial (Bunch-Kaufman) pivoting\n", path, "Symmetric")
		} else {
			fmt.Printf("\n %3s:  %9s indefinite packed matrices, partial (Bunch-Kaufman) pivoting\n", path, "Symmetric")
		}
		fmt.Printf(" Matrix types:\n")
		if sord {
			fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")
		} else {
			fmt.Printf("    1. Diagonal                        7. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              8. Random, CNDNUM = 0.1/EPS\n    3. First row and column zero       9. Scaled near underflow\n    4. Last row and column zero       10. Scaled near overflow\n    5. Middle row and column zero     11. Block diagonal matrix\n    6. Last n/2 rows and columns zero\n")
		}
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 5)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 6)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 7)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 8)
		fmt.Printf(" Messages:\n")

	} else if p2 == "HA" {
		//        HA: Hermitian,
		//            with Assen Algorithm
		fmt.Printf("\n %3s:  %9s indefinite matrices, partial (Bunch-Kaufman) pivoting\n", path, "Hermitian")

		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 5)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 6)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 7)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 8)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 9)
		fmt.Printf(" Messages:\n")

	} else if p2 == "HE" {
		//        HE: Hermitian indefinite full,
		//            with partial (Bunch-Kaufman) pivoting algorithm
		fmt.Printf("\n %3s:  %9s indefinite matrices, partial (Bunch-Kaufman) pivoting\n", path, "Hermitian")

		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 5)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 6)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 7)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 8)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 9)
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
		fmt.Printf("\n %3s:  %9s indefinite matrices, 'rook' (bounded Bunch-Kaufman) pivoting\n", path, "Hermitian")

		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")

		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: ABS( Largest element in L )\n             - ( 1 / ( 1 - ALPHA ) ) + THRESH\n", 3)
		fmt.Printf("       where ALPHA = ( 1 + SQRT( 17 ) ) / 8\n")
		fmt.Printf("   %2d: Largest 2-Norm of 2-by-2 pivots\n             - ( ( 1 + ALPHA ) / ( 1 - ALPHA ) ) + THRESH\n", 4)
		fmt.Printf("       where ALPHA = ( 1 + SQRT( 17 ) ) / 8\n")
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 6)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "HP" {
		//        HP: Hermitian indefinite packed,
		//            with partial (Bunch-Kaufman) pivoting algorithm
		if c3 == "E" {
			fmt.Printf("\n %3s:  %9s indefinite matrices, partial (Bunch-Kaufman) pivoting\n", path, "Hermitian")
		} else {
			fmt.Printf("\n %3s:  %9s indefinite packed matrices, partial (Bunch-Kaufman) pivoting\n", path, "Hermitian")
		}
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Last n/2 rows and columns zero\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n    4. Last row and column zero        9. Scaled near underflow\n    5. Middle row and column zero     10. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 5)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 6)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 7)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 8)
		fmt.Printf(" Messages:\n")

	} else if p2 == "TR" || p2 == "TP" {
		//        TR: Triangular full
		//        TP: Triangular packed
		if c3 == "R" {
			fmt.Printf("\n %3s:  Triangular matrices\n", path)
			subnam = string(path[:1]) + "LATRS"
		} else {
			fmt.Printf("\n %3s:  Triangular packed matrices\n", path)
			subnam = string(path[:1]) + "LATPS"
		}
		fmt.Printf(" Matrix types for %3s routines:\n    1. Diagonal                        6. Scaled near overflow\n    2. Random, CNDNUM = 2              7. Identity\n    3. Random, CNDNUM = sqrt(0.1/EPS)  8. Unit triangular, CNDNUM = 2\n    4. Random, CNDNUM = 0.1/EPS        9. Unit, CNDNUM = sqrt(0.1/EPS)\n    5. Scaled near underflow          10. Unit, CNDNUM = 0.1/EPS\n", path)
		fmt.Printf(" Special types for testing %s:\n   11. Matrix elements are O(1), large right hand side\n   12. First diagonal causes overflow, offdiagonal column norms < 1\n   13. First diagonal causes overflow, offdiagonal column norms > 1\n   14. Growth factor underflows, solution does not overflow\n   15. Small diagonal causes gradual overflow\n   16. One zero diagonal element\n   17. Large offdiagonals cause overflow when adding a column\n   18. Unit triangular with large right hand side\n", subnam[1:])
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 4)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 5)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 6)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 7)
		fmt.Printf(" Test ratio for %s:\n   %2d: norm( s*b - A*x )  / ( norm(A) * norm(x) * EPS )\n", subnam[1:], 8)
		fmt.Printf(" Messages:\n")

	} else if p2 == "TB" {
		//        TB: Triangular band
		fmt.Printf("\n %3s:  Triangular band matrices\n", path)
		subnam = string(path[:1]) + "LATBS"
		fmt.Printf(" Matrix types for %3s routines:\n    1. Random, CNDNUM = 2              6. Identity\n    2. Random, CNDNUM = sqrt(0.1/EPS)  7. Unit triangular, CNDNUM = 2\n    3. Random, CNDNUM = 0.1/EPS        8. Unit, CNDNUM = sqrt(0.1/EPS)\n    4. Scaled near underflow           9. Unit, CNDNUM = 0.1/EPS\n    5. Scaled near overflow\n", path)
		fmt.Printf(" Special types for testing %s:\n   10. Matrix elements are O(1), large right hand side\n   11. First diagonal causes overflow, offdiagonal column norms < 1\n   12. First diagonal causes overflow, offdiagonal column norms > 1\n   13. Growth factor underflows, solution does not overflow\n   14. Small diagonal causes gradual overflow\n   15. One zero diagonal element\n   16. Large offdiagonals cause overflow when adding a column\n   17. Unit triangular with large right hand side\n", subnam[1:])
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )\n", 2)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined\n", 3)
		fmt.Printf("   %2d: norm( X - XACT )   / ( norm(XACT) * (error bound) )\n", 4)
		fmt.Printf("   %2d: (backward error)   / EPS\n", 5)
		fmt.Printf("   %2d: RCOND * CNDNUM - 1.0\n", 6)
		fmt.Printf(" Test ratio for %s:\n   %2d: norm( s*b - A*x )  / ( norm(A) * norm(x) * EPS )\n", subnam[1:], 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "QR" {
		//        QR decomposition of rectangular matrices
		fmt.Printf("\n %3s:  %2s factorization of general matrices\n", path, "QR")
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        5. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Upper triangular                6. Random, CNDNUM = 0.1/EPS\n    3. Lower triangular                7. Scaled near underflow\n    4. Random, CNDNUM = 2              8. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( R - Q' * A ) / ( M * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( R - Q' * A ) / ( M * norm(A) * EPS )         [RFPG]\n", 8)
		fmt.Printf("   %2d: norm( I - Q'*Q )   / ( M * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C )  / ( %c * norm(C) * EPS )\n", 3, 'M')
		fmt.Printf("   %2d: norm( C*Q - C*Q )  / ( %c * norm(C) * EPS )\n", 4, 'M')
		fmt.Printf("   %2d: norm( Q'*C - Q'*C )/ ( %c * norm(C) * EPS )\n", 5, 'M')
		fmt.Printf("   %2d: norm( C*Q' - C*Q' )/ ( %c * norm(C) * EPS )\n", 6, 'M')
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 7)
		fmt.Printf("   %2d: diagonal is not non-negative\n", 9)
		fmt.Printf(" Messages:\n")

	} else if p2 == "LQ" {
		//        LQ decomposition of rectangular matrices
		fmt.Printf("\n %3s:  %2s factorization of general matrices\n", path, "LQ")
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        5. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Upper triangular                6. Random, CNDNUM = 0.1/EPS\n    3. Lower triangular                7. Scaled near underflow\n    4. Random, CNDNUM = 2              8. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L - A * Q' ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q*Q' )   / ( N * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C )  / ( %c * norm(C) * EPS )\n", 3, 'N')
		fmt.Printf("   %2d: norm( C*Q - C*Q )  / ( %c * norm(C) * EPS )\n", 4, 'N')
		fmt.Printf("   %2d: norm( Q'*C - Q'*C )/ ( %c * norm(C) * EPS )\n", 5, 'N')
		fmt.Printf("   %2d: norm( C*Q' - C*Q' )/ ( %c * norm(C) * EPS )\n", 6, 'N')
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "QL" {
		//        QL decomposition of rectangular matrices
		fmt.Printf("\n %3s:  %2s factorization of general matrices\n", path, "QL")
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        5. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Upper triangular                6. Random, CNDNUM = 0.1/EPS\n    3. Lower triangular                7. Scaled near underflow\n    4. Random, CNDNUM = 2              8. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L - Q' * A ) / ( M * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q'*Q )   / ( M * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C )  / ( %c * norm(C) * EPS )\n", 3, 'M')
		fmt.Printf("   %2d: norm( C*Q - C*Q )  / ( %c * norm(C) * EPS )\n", 4, 'M')
		fmt.Printf("   %2d: norm( Q'*C - Q'*C )/ ( %c * norm(C) * EPS )\n", 5, 'M')
		fmt.Printf("   %2d: norm( C*Q' - C*Q' )/ ( %c * norm(C) * EPS )\n", 6, 'M')
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "RQ" {
		//        RQ decomposition of rectangular matrices
		fmt.Printf("\n %3s:  %2s factorization of general matrices\n", path, "RQ")
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        5. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Upper triangular                6. Random, CNDNUM = 0.1/EPS\n    3. Lower triangular                7. Scaled near underflow\n    4. Random, CNDNUM = 2              8. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( R - A * Q' ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q*Q' )   / ( N * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C )  / ( %c * norm(C) * EPS )\n", 3, 'N')
		fmt.Printf("   %2d: norm( C*Q - C*Q )  / ( %c * norm(C) * EPS )\n", 4, 'N')
		fmt.Printf("   %2d: norm( Q'*C - Q'*C )/ ( %c * norm(C) * EPS )\n", 5, 'N')
		fmt.Printf("   %2d: norm( C*Q' - C*Q' )/ ( %c * norm(C) * EPS )\n", 6, 'N')
		fmt.Printf("   %2d: norm( B - A * X )  / ( norm(A) * norm(X) * EPS )\n", 7)
		fmt.Printf(" Messages:\n")

	} else if p2 == "QP" {
		//        QR decomposition with column pivoting
		fmt.Printf("\n %3s:  QR factorization with column pivoting\n", path)
		fmt.Printf(" Matrix types (2-6 have condition 1/EPS):\n    1. Zero matrix                     4. First n/2 columns fixed\n    2. One small eigenvalue            5. Last n/2 columns fixed\n    3. Geometric distribution          6. Every second column fixed\n")
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm(svd(A) - svd(R)) / ( M * norm(svd(R)) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( A*P - Q*R )     / ( M * norm(A) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( I - Q'*Q )      / ( M * EPS )\n", 3)
		fmt.Printf(" Messages:\n")

	} else if p2 == "TZ" {
		//        TZ:  Trapezoidal
		fmt.Printf("\n %3s:  RQ factorization of trapezoidal matrix\n", path)
		fmt.Printf(" Matrix types (2-3 have condition 1/EPS):\n    1. Zero matrix\n    2. One small eigenvalue\n    3. Geometric distribution\n")
		fmt.Printf(" Test ratios (1-3: %vTZRZF):\n", c1)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm(svd(A) - svd(R)) / ( M * norm(svd(R)) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( A - R*Q )       / ( M * norm(A) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( I - Q'*Q )      / ( M * EPS )\n", 3)
		fmt.Printf(" Messages:\n")

	} else if p2 == "LS" {
		//        LS:  Least Squares driver routines for
		//             LS, LSD, LSS, LSX and LSY.
		fmt.Printf("\n %3s:  Least squares driver routines\n", path)
		fmt.Printf(" Matrix types (1-3: full rank, 4-6: rank deficient):\n    1 and 4. Normal scaling\n    2 and 5. Scaled near overflow\n    3 and 6. Scaled near underflow\n")
		fmt.Printf(" Test ratios:\n    (1-2: %vGELS, 3-6: %vGELSY, 7-10: %vGELSS, 11-14: %vGELSD, 15-16: %vGETSLS)\n", c1, c1, c1, c1, c1)
		fmt.Printf("   %2d: norm( B - A * X )   / ( max(M,N) * norm(A) * norm(X) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( (A*X-B)' *A ) / ( max(M,N,NRHS) * norm(A) * norm(B) * EPS )\n       if TRANS='N' and M.GE.N or TRANS='T' and M.LT.N, otherwise\n       check if X is in the row space of A or A' (overdetermined case)\n", 2)
		fmt.Printf("   %2d: norm(svd(A)-svd(R)) / ( min(M,N) * norm(svd(R)) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( B - A * X )   / ( max(M,N) * norm(A) * norm(X) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( (A*X-B)' *A ) / ( max(M,N,NRHS) * norm(A) * norm(B) * EPS )\n", 5)
		fmt.Printf("   %2d: Check if X is in the row space of A or A'\n", 6)
		fmt.Printf("    7-10: same as 3-6    11-14: same as 3-6\n")
		fmt.Printf(" Messages:\n")

	} else if p2 == "LU" {
		//        LU factorization variants
		fmt.Printf("\n %3s:  LU factorization variants\n", path)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        7. Last n/2 columns zero\n    2. Upper triangular                8. Random, CNDNUM = sqrt(0.1/EPS)\n    3. Lower triangular                9. Random, CNDNUM = 0.1/EPS\n    4. Random, CNDNUM = 2             10. Scaled near underflow\n    5. First column zero              11. Scaled near overflow\n    6. Last column zero\n")
		fmt.Printf(" Test ratio:\n")
		fmt.Printf("   %2d: norm( L * U - A )  / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf(" Messages:\n")

	} else if p2 == "CH" {
		//        Cholesky factorization variants
		fmt.Printf("\n %3s:  Cholesky factorization variants\n", path)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        6. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Random, CNDNUM = 2              7. Random, CNDNUM = 0.1/EPS\n   *3. First row and column zero       8. Scaled near underflow\n   *4. Last row and column zero        9. Scaled near overflow\n   *5. Middle row and column zero\n   (* - tests error exits, no test ratios are computed)\n")
		fmt.Printf(" Test ratio:\n")
		fmt.Printf("   %2d: norm( U' * U - A ) / ( N * norm(A) * EPS ), or\n       norm( L * L' - A ) / ( N * norm(A) * EPS )\n", 1)
		fmt.Printf(" Messages:\n")

	} else if p2 == "QS" {
		//        QR factorization variants
		fmt.Printf("\n %3s:  QR factorization variants\n", path)
		fmt.Printf(" Matrix types:\n")
		fmt.Printf("    1. Diagonal                        5. Random, CNDNUM = sqrt(0.1/EPS)\n    2. Upper triangular                6. Random, CNDNUM = 0.1/EPS\n    3. Lower triangular                7. Scaled near underflow\n    4. Random, CNDNUM = 2              8. Scaled near overflow\n")
		fmt.Printf(" Test ratios:\n")

	} else if p2 == "QT" {
		//        QRT (general matrices)
		fmt.Printf("\n %3s:  QRT factorization for general matrices\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( R - Q'*A ) / ( M * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q'*Q ) / ( M * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C ) / ( M * norm(C) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( Q'*C - Q'*C ) / ( M * norm(C) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( C*Q - C*Q ) / ( M * norm(C) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( C*Q' - C*Q' ) / ( M * norm(C) * EPS )\n", 6)

	} else if p2 == "qx" {
		//        QRT (triangular-pentagonal)
		fmt.Printf("\n %3s:  QRT factorization for triangular-pentagonal matrices\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( R - Q'*A ) / ( (M+N) * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q'*Q ) / ( (M+N) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )\n", 6)

	} else if p2 == "TQ" {
		//        QRT (triangular-pentagonal)
		fmt.Printf("\n %3s:  LQT factorization for general matrices\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L - A*Q' ) / ( (M+N) * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q*Q' ) / ( (M+N) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )\n", 6)

	} else if p2 == "XQ" {
		//        QRT (triangular-pentagonal)
		fmt.Printf("\n %3s:  LQT factorization for triangular-pentagonal matrices\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( L - A*Q' ) / ( (M+N) * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q*Q' ) / ( (M+N) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )\n", 6)

	} else if p2 == "TS" {
		//        TS:  QR routines for tall-skinny and short-wide matrices
		fmt.Printf("\n %3s:  TS factorization for tall-skinny or short-wide matrices\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( R - Q'*A ) / ( (M+N) * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q'*Q ) / ( (M+N) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )\n", 6)

	} else if p2 == "HH" {
		//        HH:  Householder reconstruction for tall-skinny matrices
		fmt.Printf("\n %3s:  Householder recostruction from TSQR factorization output \n for tall-skinny matrices.\n", path)
		fmt.Printf(" Test ratios:\n")
		fmt.Printf("   %2d: norm( R - Q'*A ) / ( M * norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( I - Q'*Q ) / ( M * EPS )\n", 2)
		fmt.Printf("   %2d: norm( Q*C - Q*C ) / ( M * norm(C) * EPS )\n", 3)
		fmt.Printf("   %2d: norm( Q'*C - Q'*C ) / ( M * norm(C) * EPS )\n", 4)
		fmt.Printf("   %2d: norm( C*Q - C*Q ) / ( M * norm(C) * EPS )\n", 5)
		fmt.Printf("   %2d: norm( C*Q' - C*Q' ) / ( M * norm(C) * EPS )\n", 6)

	} else {
		//        Print error message if no header is available.
		fmt.Printf("\n %3s:  No header available\n", path)
	}

}
