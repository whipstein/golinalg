package eig

import "fmt"

// dlahd2 prints header information for the different test paths.
func dlahd2(path string) {
	var corz, sord bool

	sord = path[0] == 'S' || path[0] == 'D'
	corz = path[0] == 'C' || path[0] == 'Z'
	if !sord && !corz {
		fmt.Printf(" %3s:  no header available\n", path)
	}
	c2 := path[1:3]

	if c2 == "hs" {
		if sord {
			//           Real Non-symmetric Eigenvalue Problem:
			fmt.Printf("\n %3s -- Real Non-symmetric eigenvalue problem\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xCHKHS for details): \n")
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
			fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex %6s\n 12=Well-cond., random complex %6s    17=Ill-cond., large rand. complx %4s\n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx %4s\n", "pairs ", "pairs ", "prs.", "prs.")
			fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n")

			//           Tests performed
			fmt.Printf("\n Tests performed:   (H is Hessenberg, T is Schur, U and Z are %s,\n                    %s, W is a diagonal matrix of eigenvalues,\n                    L and R are the left and right eigenvector matrices)\n  1 = | A - U H U%c | / ( |A| n ulp )           2 = | I - U U%c | / ( n ulp )\n  3 = | H - Z T Z%c | / ( |H| n ulp )           4 = | I - Z Z%c | / ( n ulp )\n  5 = | A - UZ T (UZ)%c | / ( |A| n ulp )       6 = | I - UZ (UZ)%c | / ( n ulp )\n  7 = | T(e.vects.) - T(no e.vects.) | / ( |T| ulp )\n  8 = | W(e.vects.) - W(no e.vects.) | / ( |W| ulp )\n  9 = | TR - RW | / ( |T| |R| ulp )      10 = | LT - WL | / ( |T| |L| ulp )\n 11= |HX - XW| / (|H| |X| ulp)  (inv.it) 12= |YH - WY| / (|H| |Y| ulp)  (inv.it)\n", "orthogonal", "'=transpose", '\'', '\'', '\'', '\'', '\'', '\'')

		} else {
			//           Complex Non-symmetric Eigenvalue Problem:
			fmt.Printf("\n %3s -- Complex Non-symmetric eigenvalue problem\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xCHKHS for details): \n")
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
			fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex %6s\n 12=Well-cond., random complex %6s    17=Ill-cond., large rand. complx %4s\n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx %4s\n", "e.vals", "e.vals", "e.vs", "e.vs")
			fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n")

			//           Tests performed
			fmt.Printf("\n Tests performed:   (H is Hessenberg, T is Schur, U and Z are %s,\n                    %s, W is a diagonal matrix of eigenvalues,\n                    L and R are the left and right eigenvector matrices)\n  1 = | A - U H U%c | / ( |A| n ulp )           2 = | I - U U%c | / ( n ulp )\n  3 = | H - Z T Z%c | / ( |H| n ulp )           4 = | I - Z Z%c | / ( n ulp )\n  5 = | A - UZ T (UZ)%c | / ( |A| n ulp )       6 = | I - UZ (UZ)%c | / ( n ulp )\n  7 = | T(e.vects.) - T(no e.vects.) | / ( |T| ulp )\n  8 = | W(e.vects.) - W(no e.vects.) | / ( |W| ulp )\n  9 = | TR - RW | / ( |T| |R| ulp )      10 = | LT - WL | / ( |T| |L| ulp )\n 11= |HX - XW| / (|H| |X| ulp)  (inv.it) 12= |YH - WY| / (|H| |Y| ulp)  (inv.it)\n", "unitary", "*=conj.transp.", '*', '*', '*', '*', '*', '*')
		}

	} else if c2 == "st" {

		if sord {
			//           Real Symmetric Eigenvalue Problem:
			fmt.Printf("\n %3s -- Real Symmetric eigenvalue problem\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xDRVST for details): \n")
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
			fmt.Printf(" Dense %s Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Symmetric")

			//           Tests performed
			fmt.Printf("\n Tests performed:  See sdrvst.f\n")

		} else {
			//           Complex Hermitian Eigenvalue Problem:
			fmt.Printf("\n %3s -- Complex Hermitian eigenvalue problem\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xDRVST for details): \n")
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
			fmt.Printf(" Dense %s Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Hermitian")

			//           Tests performed
			fmt.Printf("\n Tests performed:  See cdrvst.f\n")
		}

	} else if c2 == "sg" {

		if sord {
			//           Real Symmetric Generalized Eigenvalue Problem:
			fmt.Printf("\n %3s -- Real Symmetric Generalized eigenvalue problem\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xDRVSG for details): \n")
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
			fmt.Printf(" Dense or Banded %s Matrices: \n  8=Evenly spaced eigenvals.          15=Matrix with small random entries.\n  9=Geometrically spaced eigenvals.   16=Evenly spaced eigenvals, KA=1, KB=1.\n 10=Clustered eigenvalues.            17=Evenly spaced eigenvals, KA=2, KB=1.\n 11=Large, evenly spaced eigenvals.   18=Evenly spaced eigenvals, KA=2, KB=2.\n 12=Small, evenly spaced eigenvals.   19=Evenly spaced eigenvals, KA=3, KB=1.\n 13=Matrix with random O(1) entries.  20=Evenly spaced eigenvals, KA=3, KB=2.\n 14=Matrix with large random entries. 21=Evenly spaced eigenvals, KA=3, KB=3.\n", "Symmetric")

			//           Tests performed
			fmt.Printf("\n Tests performed:   \n( For each pair (A,B), where A is of the given type \n and B is a random well-conditioned matrix. D is \n diagonal, and Z is orthogonal. )\n 1 = DSYGV, with ITYPE=1 and UPLO='U':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 2 = DSPGV, with ITYPE=1 and UPLO='U':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 3 = DSBGV, with ITYPE=1 and UPLO='U':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 4 = DSYGV, with ITYPE=1 and UPLO='L':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 5 = DSPGV, with ITYPE=1 and UPLO='L':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 6 = DSBGV, with ITYPE=1 and UPLO='L':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n")
			fmt.Printf(" 7 = DSYGV, with ITYPE=2 and UPLO='U':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n 8 = DSPGV, with ITYPE=2 and UPLO='U':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n 9 = DSPGV, with ITYPE=2 and UPLO='L':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n10 = DSPGV, with ITYPE=2 and UPLO='L':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n11 = DSYGV, with ITYPE=3 and UPLO='U':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n12 = DSPGV, with ITYPE=3 and UPLO='U':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n13 = DSYGV, with ITYPE=3 and UPLO='L':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n14 = DSPGV, with ITYPE=3 and UPLO='L':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n")

		} else {
			//           Complex Hermitian Generalized Eigenvalue Problem:
			fmt.Printf("\n %3s -- Complex Hermitian Generalized eigenvalue problem\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xDRVSG for details): \n")
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
			fmt.Printf(" Dense or Banded %s Matrices: \n  8=Evenly spaced eigenvals.          15=Matrix with small random entries.\n  9=Geometrically spaced eigenvals.   16=Evenly spaced eigenvals, KA=1, KB=1.\n 10=Clustered eigenvalues.            17=Evenly spaced eigenvals, KA=2, KB=1.\n 11=Large, evenly spaced eigenvals.   18=Evenly spaced eigenvals, KA=2, KB=2.\n 12=Small, evenly spaced eigenvals.   19=Evenly spaced eigenvals, KA=3, KB=1.\n 13=Matrix with random O(1) entries.  20=Evenly spaced eigenvals, KA=3, KB=2.\n 14=Matrix with large random entries. 21=Evenly spaced eigenvals, KA=3, KB=3.\n", "Hermitian")

			//           Tests performed
			fmt.Printf("\n Tests performed:   \n( For each pair (A,B), where A is of the given type \n and B is a random well-conditioned matrix. D is \n diagonal, and Z is unitary. )\n 1 = ZHEGV, with ITYPE=1 and UPLO='U':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 2 = ZHPGV, with ITYPE=1 and UPLO='U':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 3 = ZHBGV, with ITYPE=1 and UPLO='U':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 4 = ZHEGV, with ITYPE=1 and UPLO='L':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 5 = ZHPGV, with ITYPE=1 and UPLO='L':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n 6 = ZHBGV, with ITYPE=1 and UPLO='L':  | A Z - B Z D | / ( |A| |Z| n ulp )     \n")
			fmt.Printf(" 7 = ZHEGV, with ITYPE=2 and UPLO='U':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n 8 = ZHPGV, with ITYPE=2 and UPLO='U':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n 9 = ZHPGV, with ITYPE=2 and UPLO='L':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n10 = ZHPGV, with ITYPE=2 and UPLO='L':  | A B Z - Z D | / ( |A| |Z| n ulp )     \n11 = ZHEGV, with ITYPE=3 and UPLO='U':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n12 = ZHPGV, with ITYPE=3 and UPLO='U':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n13 = ZHEGV, with ITYPE=3 and UPLO='L':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n14 = ZHPGV, with ITYPE=3 and UPLO='L':  | B A Z - Z D | / ( |A| |Z| n ulp )     \n")

		}

	} else if c2 == "bd" {

		if sord {
			//           Real Singular Value Decomposition:
			fmt.Printf("\n %3s -- Real Singular Value Decomposition\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xCHKBD for details):\n Diagonal matrices:\n   1: Zero                             5: Clustered entries\n   2: Identity                         6: Large, evenly spaced entries\n   3: Evenly spaced entries            7: Small, evenly spaced entries\n   4: Geometrically spaced entries\n General matrices:\n   8: Evenly spaced sing. vals.       12: Small, evenly spaced sing vals\n   9: Geometrically spaced sing vals  13: Random, O(1) entries\n  10: Clustered sing. vals.           14: Random, scaled near overflow\n  11: Large, evenly spaced sing vals  15: Random, scaled near underflow\n")

			//           Tests performed
			fmt.Printf("\n Test ratios:  (B: bidiagonal, S: diagonal, Q, P, U, and V: %10s\n                X: m x nrhs, Y = Q' X, and Z = U' Y)\n", "orthogonal")
			fmt.Printf("   1: norm( A - Q B P' ) / ( norm(A) max(m,n) ulp )\n   2: norm( I - Q' Q )   / ( m ulp )\n   3: norm( I - P' P )   / ( n ulp )\n   4: norm( B - U S V' ) / ( norm(B) min(m,n) ulp )\n   5: norm( Y - U Z )    / ( norm(Z) max(min(m,n),k) ulp )\n   6: norm( I - U' U )   / ( min(m,n) ulp )\n   7: norm( I - V' V )   / ( min(m,n) ulp )\n   8: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n   9: norm( S - S1 )     / ( norm(S) ulp ), where S1 is computed\n                                            without computing U and V'\n  10: Sturm sequence test (0 if sing. vals of B within THRESH of S)\n  11: norm( A - (QU) S (V' P') ) / ( norm(A) max(m,n) ulp )\n  12: norm( X - (QU) Z )         / ( |X| max(M,k) ulp )\n  13: norm( I - (QU)'(QU) )      / ( M ulp )\n  14: norm( I - (V' P') (P V) )  / ( N ulp )\n  15: norm( B - U S V' ) / ( norm(B) min(m,n) ulp )\n  16: norm( I - U' U )   / ( min(m,n) ulp )\n  17: norm( I - V' V )   / ( min(m,n) ulp )\n  18: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  19: norm( S - S1 )     / ( norm(S) ulp ), where S1 is computed\n                                            without computing U and V'\n  20: norm( B - U S V' )  / ( norm(B) min(m,n) ulp )  DBDSVX(V,A)\n  21: norm( I - U' U )    / ( min(m,n) ulp )\n  22: norm( I - V' V )    / ( min(m,n) ulp )\n  23: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  24: norm( S - S1 )      / ( norm(S) ulp ), where S1 is computed\n                                             without computing U and V'\n  25: norm( S - U' B V ) / ( norm(B) n ulp )  DBDSVX(V,I)\n  26: norm( I - U' U )    / ( min(m,n) ulp )\n  27: norm( I - V' V )    / ( min(m,n) ulp )\n  28: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  29: norm( S - S1 )      / ( norm(S) ulp ), where S1 is computed\n                                             without computing U and V'\n  30: norm( S - U' B V ) / ( norm(B) n ulp )  DBDSVX(V,V)\n  31: norm( I - U' U )    / ( min(m,n) ulp )\n  32: norm( I - V' V )    / ( min(m,n) ulp )\n  33: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  34: norm( S - S1 )      / ( norm(S) ulp ), where S1 is computed\n                                             without computing U and V'\n")
		} else {
			//           Complex Singular Value Decomposition:
			fmt.Printf("\n %3s -- Complex Singular Value Decomposition\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xCHKBD for details):\n Diagonal matrices:\n   1: Zero                             5: Clustered entries\n   2: Identity                         6: Large, evenly spaced entries\n   3: Evenly spaced entries            7: Small, evenly spaced entries\n   4: Geometrically spaced entries\n General matrices:\n   8: Evenly spaced sing. vals.       12: Small, evenly spaced sing vals\n   9: Geometrically spaced sing vals  13: Random, O(1) entries\n  10: Clustered sing. vals.           14: Random, scaled near overflow\n  11: Large, evenly spaced sing vals  15: Random, scaled near underflow\n")

			//           Tests performed
			fmt.Printf("\n Test ratios:  (B: bidiagonal, S: diagonal, Q, P, U, and V: %10s\n                X: m x nrhs, Y = Q' X, and Z = U' Y)\n", "unitary   ")
			fmt.Printf("   1: norm( A - Q B P' ) / ( norm(A) max(m,n) ulp )\n   2: norm( I - Q' Q )   / ( m ulp )\n   3: norm( I - P' P )   / ( n ulp )\n   4: norm( B - U S V' ) / ( norm(B) min(m,n) ulp )\n   5: norm( Y - U Z )    / ( norm(Z) max(min(m,n),k) ulp )\n   6: norm( I - U' U )   / ( min(m,n) ulp )\n   7: norm( I - V' V )   / ( min(m,n) ulp )\n   8: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n   9: norm( S - S1 )     / ( norm(S) ulp ), where S1 is computed\n                                            without computing U and V'\n  10: Sturm sequence test (0 if sing. vals of B within THRESH of S)\n  11: norm( A - (QU) S (V' P') ) / ( norm(A) max(m,n) ulp )\n  12: norm( X - (QU) Z )         / ( |X| max(M,k) ulp )\n  13: norm( I - (QU)'(QU) )      / ( M ulp )\n  14: norm( I - (V' P') (P V) )  / ( N ulp )\n  15: norm( B - U S V' ) / ( norm(B) min(m,n) ulp )\n  16: norm( I - U' U )   / ( min(m,n) ulp )\n  17: norm( I - V' V )   / ( min(m,n) ulp )\n  18: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  19: norm( S - S1 )     / ( norm(S) ulp ), where S1 is computed\n                                            without computing U and V'\n  20: norm( B - U S V' )  / ( norm(B) min(m,n) ulp )  DBDSVX(V,A)\n  21: norm( I - U' U )    / ( min(m,n) ulp )\n  22: norm( I - V' V )    / ( min(m,n) ulp )\n  23: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  24: norm( S - S1 )      / ( norm(S) ulp ), where S1 is computed\n                                             without computing U and V'\n  25: norm( S - U' B V ) / ( norm(B) n ulp )  DBDSVX(V,I)\n  26: norm( I - U' U )    / ( min(m,n) ulp )\n  27: norm( I - V' V )    / ( min(m,n) ulp )\n  28: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  29: norm( S - S1 )      / ( norm(S) ulp ), where S1 is computed\n                                             without computing U and V'\n  30: norm( S - U' B V ) / ( norm(B) n ulp )  DBDSVX(V,V)\n  31: norm( I - U' U )    / ( min(m,n) ulp )\n  32: norm( I - V' V )    / ( min(m,n) ulp )\n  33: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n  34: norm( S - S1 )      / ( norm(S) ulp ), where S1 is computed\n                                             without computing U and V'\n")
		}

	} else if c2 == "bb" {

		if sord {
			//           Real General Band reduction to bidiagonal form:
			fmt.Printf("\n %3s -- Real Band reduc. to bidiagonal form\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xCHKBB for details):\n Diagonal matrices:\n   1: Zero                             5: Clustered entries\n   2: Identity                         6: Large, evenly spaced entries\n   3: Evenly spaced entries            7: Small, evenly spaced entries\n   4: Geometrically spaced entries\n General matrices:\n   8: Evenly spaced sing. vals.       12: Small, evenly spaced sing vals\n   9: Geometrically spaced sing vals  13: Random, O(1) entries\n  10: Clustered sing. vals.           14: Random, scaled near overflow\n  11: Large, evenly spaced sing vals  15: Random, scaled near underflow\n")

			//           Tests performed
			fmt.Printf("\n Test ratios:  (B: upper bidiagonal, Q and P: %10s\n                C: m x nrhs, PT = P', Y = Q' C)\n 1: norm( A - Q B PT ) / ( norm(A) max(m,n) ulp )\n 2: norm( I - Q' Q )   / ( m ulp )\n 3: norm( I - PT PT' )   / ( n ulp )\n 4: norm( Y - Q' C )   / ( norm(Y) max(m,nrhs) ulp )\n", "orthogonal")
		} else {
			//           Complex Band reduction to bidiagonal form:
			fmt.Printf("\n %3s -- Complex Band reduc. to bidiagonal form\n", path)

			//           Matrix types
			fmt.Printf(" Matrix types (see xCHKBB for details):\n Diagonal matrices:\n   1: Zero                             5: Clustered entries\n   2: Identity                         6: Large, evenly spaced entries\n   3: Evenly spaced entries            7: Small, evenly spaced entries\n   4: Geometrically spaced entries\n General matrices:\n   8: Evenly spaced sing. vals.       12: Small, evenly spaced sing vals\n   9: Geometrically spaced sing vals  13: Random, O(1) entries\n  10: Clustered sing. vals.           14: Random, scaled near overflow\n  11: Large, evenly spaced sing vals  15: Random, scaled near underflow\n")

			//           Tests performed
			fmt.Printf("\n Test ratios:  (B: upper bidiagonal, Q and P: %10s\n                C: m x nrhs, PT = P', Y = Q' C)\n 1: norm( A - Q B PT ) / ( norm(A) max(m,n) ulp )\n 2: norm( I - Q' Q )   / ( m ulp )\n 3: norm( I - PT PT' )   / ( n ulp )\n 4: norm( Y - Q' C )   / ( norm(Y) max(m,nrhs) ulp )\n", "unitary   ")
		}

	} else {

		fmt.Printf(" %3s:  no header available\n", path)
		return
	}
	//
	//
	//     Symmetric/Hermitian eigenproblem
	//
	//
	//
	//     Symmetric/Hermitian Generalized eigenproblem
	//
	//
	//
	//     Singular Value Decomposition
	//
	//
	//
	//     Band reduction to bidiagonal form
	//
	//
	//
	//     End of DLAHD2
	//
}
