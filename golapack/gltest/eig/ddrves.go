package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"testing"
)

// Ddrves checks the nonsymmetric eigenvalue (Schur form) problem
//    driver DGEES.
//
//    When DDRVES is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified.  For each size ("n")
//    and each type of matrix, one matrix will be generated and used
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
//    (4)     0     if WR+math.Sqrt(-1)*WI are eigenvalues of T
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
//    (10)    0     if WR+math.Sqrt(-1)*WI are eigenvalues of T
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
//    if DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
//    Currently, the list of possible types is:
//
//    (1)  The zero matrix.
//    (2)  The identity matrix.
//    (3)  A (transposed) Jordan block, with 1's on the diagonal.
//
//    (4)  A diagonal matrix with evenly spaced entries
//         1, ..., ULP  and random signs.
//         (ULP = (first number larger than 1) - 1 )
//    (5)  A diagonal matrix with geometrically spaced entries
//         1, ..., ULP  and random signs.
//    (6)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//         and random signs.
//
//    (7)  Same as (4), but multiplied by a constant near
//         the overflow threshold
//    (8)  Same as (4), but multiplied by a constant near
//         the underflow threshold
//
//    (9)  A matrix of the form  U' T U, where U is orthogonal and
//         T has evenly spaced entries 1, ..., ULP with random signs
//         on the diagonal and random O(1) entries in the upper
//         triangle.
//
//    (10) A matrix of the form  U' T U, where U is orthogonal and
//         T has geometrically spaced entries 1, ..., ULP with random
//         signs on the diagonal and random O(1) entries in the upper
//         triangle.
//
//    (11) A matrix of the form  U' T U, where U is orthogonal and
//         T has "clustered" entries 1, ULP,..., ULP with random
//         signs on the diagonal and random O(1) entries in the upper
//         triangle.
//
//    (12) A matrix of the form  U' T U, where U is orthogonal and
//         T has real or complex conjugate paired eigenvalues randomly
//         chosen from ( ULP, 1 ) and random O(1) entries in the upper
//         triangle.
//
//    (13) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has evenly spaced entries 1, ..., ULP
//         with random signs on the diagonal and random O(1) entries
//         in the upper triangle.
//
//    (14) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has geometrically spaced entries
//         1, ..., ULP with random signs on the diagonal and random
//         O(1) entries in the upper triangle.
//
//    (15) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has "clustered" entries 1, ULP,..., ULP
//         with random signs on the diagonal and random O(1) entries
//         in the upper triangle.
//
//    (16) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has real or complex conjugate paired
//         eigenvalues randomly chosen from ( ULP, 1 ) and random
//         O(1) entries in the upper triangle.
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
func Ddrves(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.Matrix, lda *int, h, ht *mat.Matrix, wr, wi, wrt, wit *mat.Vector, vs *mat.Matrix, ldvs *int, result, work *mat.Vector, nwork *int, iwork *[]int, bwork *[]bool, info *int, t *testing.T) {
	var badnn bool
	var sort byte
	var anorm, cond, conds, one, ovfl, rtulp, rtulpi, tmp, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, isort, itype, iwk, j, jcol, jsize, jtype, knteig, lwork, maxtyp, mtypes, n, nerrs, nfail, nmax, nnwork, ntest, ntestf, ntestt, rsub, sdim int

	adumma := make([]byte, 1)
	res := vf(2)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kconds := make([]int, 21)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	zero = 0.0
	one = 1.0
	maxtyp = 21

	selopt := &gltest.Common.Sslct.Selopt

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17], ktype[18], ktype[19], ktype[20] = 1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17], kmagn[18], kmagn[19], kmagn[20] = 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17], kmode[18], kmode[19], kmode[20] = 0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1
	kconds[0], kconds[1], kconds[2], kconds[3], kconds[4], kconds[5], kconds[6], kconds[7], kconds[8], kconds[9], kconds[10], kconds[11], kconds[12], kconds[13], kconds[14], kconds[15], kconds[16], kconds[17], kconds[18], kconds[19], kconds[20] = 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0

	path := []byte("DES")

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
		(*info) = -17
	} else if 5*nmax+2*int(math.Pow(float64(nmax), 2)) > (*nwork) {
		(*info) = -20
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DDRVES"), -(*info))
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
		mtypes = maxtyp
		if (*nsizes) == 1 && (*ntypes) == maxtyp+1 {
			mtypes = mtypes + 1
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label260
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

			golapack.Dlaset('F', lda, &n, &zero, &zero, a, lda)
			iinfo = 0
			cond = ulpinv

			//           Special Matrices -- Identity & Jordan block
			//
			//              Zero
			if itype == 1 {
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jcol = 1; jcol <= n; jcol++ {
					a.Set(jcol-1, jcol-1, anorm)
				}

			} else if itype == 3 {
				//              Jordan Block
				for jcol = 1; jcol <= n; jcol++ {
					a.Set(jcol-1, jcol-1, anorm)
					if jcol > 1 {
						a.Set(jcol-1, jcol-1-1, one)
					}
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 6 {
				//              General, eigenvalues specified
				if kconds[jtype-1] == 1 {
					conds = one
				} else if kconds[jtype-1] == 2 {
					conds = rtulpi
				} else {
					conds = zero
				}

				adumma[0] = ' '
				matgen.Dlatme(&n, 'S', iseed, work, &imode, &cond, &one, adumma, 'T', 'T', 'T', work.Off(n+1-1), func() *int { y := 4; return &y }(), &conds, &n, &n, &anorm, a, lda, work.Off(2*n+1-1), &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              General, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)
				if n >= 4 {
					golapack.Dlaset('F', func() *int { y := 2; return &y }(), &n, &zero, &zero, a, lda)
					golapack.Dlaset('F', toPtr(n-3), func() *int { y := 1; return &y }(), &zero, &zero, a.Off(2, 0), lda)
					golapack.Dlaset('F', toPtr(n-3), func() *int { y := 2; return &y }(), &zero, &zero, a.Off(2, n-1-1), lda)
					golapack.Dlaset('F', func() *int { y := 1; return &y }(), &n, &zero, &zero, a.Off(n-1, 0), lda)
				}

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DDRVES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
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
					nnwork = 5*n + 2*int(math.Pow(float64(n), 2))
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
					golapack.Dlacpy('F', &n, &n, a, lda, h, lda)
					golapack.Dgees('V', sort, Dslect, &n, h, lda, &sdim, wr, wi, vs, ldvs, work, &nnwork, bwork, &iinfo)
					if iinfo != 0 && iinfo != n+2 {
						t.Fail()
						result.Set(1+rsub-1, ulpinv)
						fmt.Printf(" DDRVES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEES1", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
						goto label220
					}

					//                 Do Test (1) or Test (7)
					result.Set(1+rsub-1, zero)
					for j = 1; j <= n-2; j++ {
						for i = j + 2; i <= n; i++ {
							if h.Get(i-1, j-1) != zero {
								result.Set(1+rsub-1, ulpinv)
							}
						}
					}
					for i = 1; i <= n-2; i++ {
						if h.Get(i+1-1, i-1) != zero && h.Get(i+2-1, i+1-1) != zero {
							result.Set(1+rsub-1, ulpinv)
						}
					}
					for i = 1; i <= n-1; i++ {
						if h.Get(i+1-1, i-1) != zero {
							if h.Get(i-1, i-1) != h.Get(i+1-1, i+1-1) || h.Get(i-1, i+1-1) == zero || math.Copysign(one, h.Get(i+1-1, i-1)) == math.Copysign(one, h.Get(i-1, i+1-1)) {
								result.Set(1+rsub-1, ulpinv)
							}
						}
					}

					//                 Do Tests (2) and (3) or Tests (8) and (9)
					lwork = maxint(1, 2*n*n)
					Dhst01(&n, func() *int { y := 1; return &y }(), &n, a, lda, h, lda, vs, ldvs, work, &lwork, res)
					result.Set(2+rsub-1, res.Get(0))
					result.Set(3+rsub-1, res.Get(1))

					//                 Do Test (4) or Test (10)
					result.Set(4+rsub-1, zero)
					for i = 1; i <= n; i++ {
						if h.Get(i-1, i-1) != wr.Get(i-1) {
							result.Set(4+rsub-1, ulpinv)
						}
					}
					if n > 1 {
						if h.Get(1, 0) == zero && wi.Get(0) != zero {
							result.Set(4+rsub-1, ulpinv)
						}
						if h.Get(n-1, n-1-1) == zero && wi.Get(n-1) != zero {
							result.Set(4+rsub-1, ulpinv)
						}
					}
					for i = 1; i <= n-1; i++ {
						if h.Get(i+1-1, i-1) != zero {
							tmp = math.Sqrt(math.Abs(h.Get(i+1-1, i-1))) * math.Sqrt(math.Abs(h.Get(i-1, i+1-1)))
							result.Set(4+rsub-1, maxf64(result.Get(4+rsub-1), math.Abs(wi.Get(i-1)-tmp)/maxf64(ulp*tmp, unfl)))
							result.Set(4+rsub-1, maxf64(result.Get(4+rsub-1), math.Abs(wi.Get(i+1-1)+tmp)/maxf64(ulp*tmp, unfl)))
						} else if i > 1 {
							if h.Get(i+1-1, i-1) == zero && h.Get(i-1, i-1-1) == zero && wi.Get(i-1) != zero {
								result.Set(4+rsub-1, ulpinv)
							}
						}
					}

					//                 Do Test (5) or Test (11)
					golapack.Dlacpy('F', &n, &n, a, lda, ht, lda)
					golapack.Dgees('N', sort, Dslect, &n, ht, lda, &sdim, wrt, wit, vs, ldvs, work, &nnwork, bwork, &iinfo)
					if iinfo != 0 && iinfo != n+2 {
						t.Fail()
						result.Set(5+rsub-1, ulpinv)
						fmt.Printf(" DDRVES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEES2", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
						goto label220
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
						if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
							result.Set(6+rsub-1, ulpinv)
						}
					}

					//                 Do Test (13)
					if isort == 1 {
						result.Set(12, zero)
						knteig = 0
						for i = 1; i <= n; i++ {
							if Dslect(wr.GetPtr(i-1), wi.GetPtr(i-1)) || Dslect(wr.GetPtr(i-1), toPtrf64(-wi.Get(i-1))) {
								knteig = knteig + 1
							}
							if i < n {
								if (Dslect(wr.GetPtr(i+1-1), wi.GetPtr(i+1-1)) || Dslect(wr.GetPtr(i+1-1), toPtrf64(-wi.Get(i+1-1)))) && (!(Dslect(wr.GetPtr(i-1), wi.GetPtr(i-1)) || Dslect(wr.GetPtr(i-1), toPtrf64(-wi.Get(i-1))))) && iinfo != n+2 {
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
			label220:
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
					t.Fail()
					ntestf = ntestf + 1
				}
				if ntestf == 1 {
					fmt.Printf("\n %3s -- Real Schur Form Decomposition Driver\n Matrix types (see DDRVES for details): \n", path)
					fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
					fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
					fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n\n")
					fmt.Printf(" Tests performed with test threshold =%8.2f\n ( A denotes A on input and T denotes A on output)\n\n 1 = 0 if T in Schur form (no sort),   1/ulp otherwise\n 2 = | A - VS T transpose(VS) | / ( n |A| ulp ) (no sort)\n 3 = | I - VS transpose(VS) | / ( n ulp ) (no sort) \n 4 = 0 if WR+math.Sqrt(-1)*WI are eigenvalues of T (no sort),  1/ulp otherwise\n 5 = 0 if T same no matter if VS computed (no sort),  1/ulp otherwise\n 6 = 0 if WR, WI same no matter if VS computed (no sort),  1/ulp otherwise\n", *thresh)
					fmt.Printf(" 7 = 0 if T in Schur form (sort),   1/ulp otherwise\n 8 = | A - VS T transpose(VS) | / ( n |A| ulp ) (sort)\n 9 = | I - VS transpose(VS) | / ( n ulp ) (sort) \n 10 = 0 if WR+math.Sqrt(-1)*WI are eigenvalues of T (sort),  1/ulp otherwise\n 11 = 0 if T same no matter if VS computed (sort),  1/ulp otherwise\n 12 = 0 if WR, WI same no matter if VS computed (sort),  1/ulp otherwise\n 13 = 0 if sorting successful, 1/ulp otherwise\n\n")
					ntestf = 2
				}

				for j = 1; j <= 13; j++ {
					if result.Get(j-1) >= (*thresh) {
						t.Fail()
						fmt.Printf(" N=%5d, IWK=%2d, seed=%4d, type %2d, test(%2d)=%10.3f\n", n, iwk, ioldsd, jtype, j, result.Get(j-1))
					}
				}

				nerrs = nerrs + nfail
				ntestt = ntestt + ntest

			}
		label260:
		}
	}

	//     Summary
	Dlasum(path, &nerrs, &ntestt)
}
