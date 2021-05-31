package eig

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zdrves checks the nonsymmetric eigenvalue (Schur form) problem
//    driver ZGEES.
//
//    When ZDRVES is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified.  For each size ("n")
//    and each _type of matrix, one matrix will be generated and used
//    to test the nonsymmetric eigenroutines.  For each matrix, 13
//    tests will be performed:
//
//    (1)     0 if T is in Schur form, 1/ulp otherwise
//           (no sorting of eigenvalues)
//
//    (2)     | A - VS T VS' | / ( n |A| ulp )
//
//      Here VS is the matrix of Schur eigenvectors, and T is in Schur
//      form  (no sorting of eigenvalues).
//
//    (3)     | I - VS VS' | / ( n ulp ) (no sorting of eigenvalues).
//
//    (4)     0     if W are eigenvalues of T
//            1/ulp otherwise
//            (no sorting of eigenvalues)
//
//    (5)     0     if T(with VS) = T(without VS),
//            1/ulp otherwise
//            (no sorting of eigenvalues)
//
//    (6)     0     if eigenvalues(with VS) = eigenvalues(without VS),
//            1/ulp otherwise
//            (no sorting of eigenvalues)
//
//    (7)     0 if T is in Schur form, 1/ulp otherwise
//            (with sorting of eigenvalues)
//
//    (8)     | A - VS T VS' | / ( n |A| ulp )
//
//      Here VS is the matrix of Schur eigenvectors, and T is in Schur
//      form  (with sorting of eigenvalues).
//
//    (9)     | I - VS VS' | / ( n ulp ) (with sorting of eigenvalues).
//
//    (10)    0     if W are eigenvalues of T
//            1/ulp otherwise
//            (with sorting of eigenvalues)
//
//    (11)    0     if T(with VS) = T(without VS),
//            1/ulp otherwise
//            (with sorting of eigenvalues)
//
//    (12)    0     if eigenvalues(with VS) = eigenvalues(without VS),
//            1/ulp otherwise
//            (with sorting of eigenvalues)
//
//    (13)    if sorting worked and SDIM is the number of
//            eigenvalues which were SELECTed
//
//    The "sizes" are specified by an array NN(1:NSIZES); the value of
//    each element NN(j) specifies one size.
//    The "types" are specified by a logical array DOTYPE( 1:NTYPES );
//    if DOTYPE(j) is .TRUE., then matrix _type "j" will be generated.
//    Currently, the list of possible types is:
//
//    (1)  The zero matrix.
//    (2)  The identity matrix.
//    (3)  A (transposed) Jordan block, with 1's on the diagonal.
//
//    (4)  A diagonal matrix with evenly spaced entries
//         1, ..., ULP  and random complex angles.
//         (ULP = (first number larger than 1) - 1 )
//    (5)  A diagonal matrix with geometrically spaced entries
//         1, ..., ULP  and random complex angles.
//    (6)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//         and random complex angles.
//
//    (7)  Same as (4), but multiplied by a constant near
//         the overflow threshold
//    (8)  Same as (4), but multiplied by a constant near
//         the underflow threshold
//
//    (9)  A matrix of the form  U' T U, where U is unitary and
//         T has evenly spaced entries 1, ..., ULP with random
//         complex angles on the diagonal and random O(1) entries in
//         the upper triangle.
//
//    (10) A matrix of the form  U' T U, where U is unitary and
//         T has geometrically spaced entries 1, ..., ULP with random
//         complex angles on the diagonal and random O(1) entries in
//         the upper triangle.
//
//    (11) A matrix of the form  U' T U, where U is orthogonal and
//         T has "clustered" entries 1, ULP,..., ULP with random
//         complex angles on the diagonal and random O(1) entries in
//         the upper triangle.
//
//    (12) A matrix of the form  U' T U, where U is unitary and
//         T has complex eigenvalues randomly chosen from
//         ULP < |z| < 1   and random O(1) entries in the upper
//         triangle.
//
//    (13) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has evenly spaced entries 1, ..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (14) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has geometrically spaced entries
//         1, ..., ULP with random complex angles on the diagonal
//         and random O(1) entries in the upper triangle.
//
//    (15) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has "clustered" entries 1, ULP,..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (16) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has complex eigenvalues randomly chosen
//         from ULP < |z| < 1 and random O(1) entries in the upper
//         triangle.
//
//    (17) Same as (16), but multiplied by a constant
//         near the overflow threshold
//    (18) Same as (16), but multiplied by a constant
//         near the underflow threshold
//
//    (19) Nonsymmetric matrix with random entries chosen from (-1,1).
//         If N is at least 4, all entries in first two rows and last
//         row, and first column and last two columns are zero.
//    (20) Same as (19), but multiplied by a constant
//         near the overflow threshold
//    (21) Same as (19), but multiplied by a constant
//         near the underflow threshold
func Zdrves(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.CMatrix, lda *int, h, ht *mat.CMatrix, w, wt *mat.CVector, vs *mat.CMatrix, ldvs *int, result *mat.Vector, work *mat.CVector, nwork *int, rwork *mat.Vector, iwork *[]int, bwork *[]bool, info *int, t *testing.T) {
	var badnn bool
	var sort byte
	var cone, czero complex128
	var anorm, cond, conds, one, ovfl, rtulp, rtulpi, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, isort, itype, iwk, j, jcol, jsize, jtype, knteig, lwork, maxtyp, mtypes, n, nerrs, nfail, nmax, nnwork, ntest, ntestf, ntestt, rsub, sdim int
	res := vf(2)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kconds := make([]int, 21)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	maxtyp = 21
	selopt := &gltest.Common.Sslct.Selopt

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17], ktype[18], ktype[19], ktype[20] = 1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17], kmagn[18], kmagn[19], kmagn[20] = 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17], kmode[18], kmode[19], kmode[20] = 0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1
	kconds[0], kconds[1], kconds[2], kconds[3], kconds[4], kconds[5], kconds[6], kconds[7], kconds[8], kconds[9], kconds[10], kconds[11], kconds[12], kconds[13], kconds[14], kconds[15], kconds[16], kconds[17], kconds[18], kconds[19], kconds[20] = 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0

	path := []byte("ZES")

	//     Check for errors
	ntestt = 0
	ntestf = 0
	(*info) = 0
	(*selopt) = 0

	//     Important constants
	badnn = false
	nmax = 0
	for j = 1; j <= (*nsizes); j++ {
		nmax = maxint(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
	}

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badnn {
		(*info) = -2
	} else if (*ntypes) < 0 {
		(*info) = -3
	} else if (*thresh) < zero {
		(*info) = -6
	} else if (*nounit) <= 0 {
		(*info) = -7
	} else if (*lda) < 1 || (*lda) < nmax {
		(*info) = -9
	} else if (*ldvs) < 1 || (*ldvs) < nmax {
		(*info) = -15
	} else if 5*nmax+2*powint(nmax, 2) > (*nwork) {
		(*info) = -18
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZDRVES"), -(*info))
		return
	}

	//     Quick return if nothing to do
	if (*nsizes) == 0 || (*ntypes) == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	golapack.Dlabad(&unfl, &ovfl)
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	rtulp = math.Sqrt(ulp)
	rtulpi = one / rtulp

	//     Loop over sizes, types
	nerrs = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		n = (*nn)[jsize-1]
		if (*nsizes) != 1 {
			mtypes = minint(maxtyp, *ntypes)
		} else {
			mtypes = minint(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label230
			}

			//           Save ISEED in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			//           Compute "A"
			//
			//           Control parameters:
			//
			//           KMAGN  KCONDS  KMODE        KTYPE
			//       =1  O(1)   1       clustered 1  zero
			//       =2  large  large   clustered 2  identity
			//       =3  small          exponential  Jordan
			//       =4                 arithmetic   diagonal, (w/ eigenvalues)
			//       =5                 random log   symmetric, w/ eigenvalues
			//       =6                 random       general, w/ eigenvalues
			//       =7                              random diagonal
			//       =8                              random symmetric
			//       =9                              random general
			//       =10                             random triangular
			if mtypes > maxtyp {
				goto label90
			}

			itype = ktype[jtype-1]
			imode = kmode[jtype-1]

			//           Compute norm
			switch kmagn[jtype-1] {
			case 1:
				goto label30
			case 2:
				goto label40
			case 3:
				goto label50
			}

		label30:
			;
			anorm = one
			goto label60

		label40:
			;
			anorm = ovfl * ulp
			goto label60

		label50:
			;
			anorm = unfl * ulpinv
			goto label60

		label60:
			;

			golapack.Zlaset('F', lda, &n, &czero, &czero, a, lda)
			iinfo = 0
			cond = ulpinv

			//           Special Matrices -- Identity & Jordan block
			if itype == 1 {
				//              Zero
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jcol = 1; jcol <= n; jcol++ {
					a.SetRe(jcol-1, jcol-1, anorm)
				}

			} else if itype == 3 {
				//              Jordan Block
				for jcol = 1; jcol <= n; jcol++ {
					a.SetRe(jcol-1, jcol-1, anorm)
					if jcol > 1 {
						a.Set(jcol-1, jcol-1-1, cone)
					}
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 6 {
				//              General, eigenvalues specified
				if kconds[jtype-1] == 1 {
					conds = one
				} else if kconds[jtype-1] == 2 {
					conds = rtulpi
				} else {
					conds = zero
				}

				matgen.Zlatme(&n, 'D', iseed, work, &imode, &cond, &cone, 'T', 'T', 'T', rwork, func() *int { y := 4; return &y }(), &conds, &n, &n, &anorm, a, lda, work.Off(2*n+1-1), &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              General, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)
				if n >= 4 {
					golapack.Zlaset('F', func() *int { y := 2; return &y }(), &n, &czero, &czero, a, lda)
					golapack.Zlaset('F', toPtr(n-3), func() *int { y := 1; return &y }(), &czero, &czero, a.Off(2, 0), lda)
					golapack.Zlaset('F', toPtr(n-3), func() *int { y := 2; return &y }(), &czero, &czero, a.Off(2, n-1-1), lda)
					golapack.Zlaset('F', func() *int { y := 1; return &y }(), &n, &czero, &czero, a.Off(n-1, 0), lda)
				}

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZDRVES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				return
			}

		label90:
			;

			//           Test for minimal and generous workspace
			for iwk = 1; iwk <= 2; iwk++ {
				if iwk == 1 {
					nnwork = 3 * n
				} else {
					nnwork = 5*n + 2*powint(n, 2)
				}
				nnwork = maxint(nnwork, 1)

				//              Initialize RESULT
				for j = 1; j <= 13; j++ {
					result.Set(j-1, -one)
				}

				//              Test with and without sorting of eigenvalues
				for isort = 0; isort <= 1; isort++ {
					if isort == 0 {
						sort = 'N'
						rsub = 0
					} else {
						sort = 'S'
						rsub = 6
					}

					//                 Compute Schur form and Schur vectors, and test them
					golapack.Zlacpy('F', &n, &n, a, lda, h, lda)
					golapack.Zgees('V', sort, Zslect, &n, h, lda, &sdim, w, vs, ldvs, work, &nnwork, rwork, bwork, &iinfo)
					if iinfo != 0 {
						t.Fail()
						result.Set(1+rsub-1, ulpinv)
						fmt.Printf(" ZDRVES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEES1", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
						goto label190
					}

					//                 Do Test (1) or Test (7)
					result.Set(1+rsub-1, zero)
					for j = 1; j <= n-1; j++ {
						for i = j + 1; i <= n; i++ {
							if h.GetRe(i-1, j-1) != zero {
								result.Set(1+rsub-1, ulpinv)
							}
						}
					}

					//                 Do Tests (2) and (3) or Tests (8) and (9)
					lwork = maxint(1, 2*n*n)
					Zhst01(&n, func() *int { y := 1; return &y }(), &n, a, lda, h, lda, vs, ldvs, work, &lwork, rwork, res)
					result.Set(2+rsub-1, res.Get(0))
					result.Set(3+rsub-1, res.Get(1))

					//                 Do Test (4) or Test (10)
					result.Set(4+rsub-1, zero)
					for i = 1; i <= n; i++ {
						if h.Get(i-1, i-1) != w.Get(i-1) {
							result.Set(4+rsub-1, ulpinv)
						}
					}

					//                 Do Test (5) or Test (11)
					golapack.Zlacpy('F', &n, &n, a, lda, ht, lda)
					golapack.Zgees('N', sort, Zslect, &n, ht, lda, &sdim, wt, vs, ldvs, work, &nnwork, rwork, bwork, &iinfo)
					if iinfo != 0 {
						t.Fail()
						result.Set(5+rsub-1, ulpinv)
						fmt.Printf(" ZDRVES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEES2", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
						goto label190
					}

					result.Set(5+rsub-1, zero)
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
								result.Set(5+rsub-1, ulpinv)
							}
						}
					}

					//                 Do Test (6) or Test (12)
					result.Set(6+rsub-1, zero)
					for i = 1; i <= n; i++ {
						if w.Get(i-1) != wt.Get(i-1) {
							result.Set(6+rsub-1, ulpinv)
						}
					}

					//                 Do Test (13)
					if isort == 1 {
						result.Set(12, zero)
						knteig = 0
						for i = 1; i <= n; i++ {
							if Zslect(w.Get(i - 1)) {
								knteig = knteig + 1
							}
							if i < n {
								if Zslect(w.Get(i+1-1)) && (!Zslect(w.Get(i - 1))) {
									result.Set(12, ulpinv)
								}
							}
						}
						if sdim != knteig {
							result.Set(12, ulpinv)
						}
					}

				}

				//              End of Loop -- Check for RESULT(j) > THRESH
			label190:
				;

				ntest = 0
				nfail = 0
				for j = 1; j <= 13; j++ {
					if result.Get(j-1) >= zero {
						ntest = ntest + 1
					}
					if result.Get(j-1) >= (*thresh) {
						nfail = nfail + 1
					}
				}

				if nfail > 0 {
					ntestf = ntestf + 1
				}
				if ntestf == 1 {
					fmt.Printf("\n %3s -- Complex Schur Form Decomposition Driver\n Matrix types (see ZDRVES for details): \n", path)
					fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
					fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex       \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx     \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx     \n")
					fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n\n")
					fmt.Printf(" Tests performed with test threshold =%8.2f\n ( A denotes A on input and T denotes A on output)\n\n 1 = 0 if T in Schur form (no sort),   1/ulp otherwise\n 2 = | A - VS T transpose(VS) | / ( n |A| ulp ) (no sort)\n 3 = | I - VS transpose(VS) | / ( n ulp ) (no sort) \n 4 = 0 if W are eigenvalues of T (no sort),  1/ulp otherwise\n 5 = 0 if T same no matter if VS computed (no sort),  1/ulp otherwise\n 6 = 0 if W same no matter if VS computed (no sort),  1/ulp otherwise\n", *thresh)
					fmt.Printf(" 7 = 0 if T in Schur form (sort),   1/ulp otherwise\n 8 = | A - VS T transpose(VS) | / ( n |A| ulp ) (sort)\n 9 = | I - VS transpose(VS) | / ( n ulp ) (sort) \n 10 = 0 if W are eigenvalues of T (sort),  1/ulp otherwise\n 11 = 0 if T same no matter if VS computed (sort),  1/ulp otherwise\n 12 = 0 if W same no matter if VS computed (sort),  1/ulp otherwise\n 13 = 0 if sorting successful, 1/ulp otherwise\n\n")
					ntestf = 2
				}

				for j = 1; j <= 13; j++ {
					if result.Get(j-1) >= (*thresh) {
						t.Fail()
						fmt.Printf(" N=%5d, IWK=%2d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", n, iwk, ioldsd, jtype, j, result.Get(j-1))
					}
				}

				nerrs = nerrs + nfail
				ntestt = ntestt + ntest

			}
		label230:
		}
	}

	//     Summary
	Dlasum(path, &nerrs, &ntestt)
}
