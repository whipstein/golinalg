package eig

import (
	"fmt"
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"math/cmplx"
	"testing"
)

// Zchkhbstg tests the reduction of a Hermitian band matrix to tridiagonal
// from, used with the Hermitian eigenvalue problem.
//
// ZHBTRD factors a Hermitian band matrix A as  U S U* , where * means
// conjugate transpose, S is symmetric tridiagonal, and U is unitary.
// ZHBTRD can use either just the lower or just the upper triangle
// of A; ZCHKHBSTG checks both cases.
//
// ZHETRD_HB2ST factors a Hermitian band matrix A as  U S U* ,
// where * means conjugate transpose, S is symmetric tridiagonal, and U is
// unitary. ZHETRD_HB2ST can use either just the lower or just
// the upper triangle of A; ZCHKHBSTG checks both cases.
//
// DSTEQR factors S as  Z D1 Z'.
// D1 is the matrix of eigenvalues computed when Z is not computed
// and from the S resulting of DSBTRD "U" (used as reference for DSYTRD_SB2ST)
// D2 is the matrix of eigenvalues computed when Z is not computed
// and from the S resulting of DSYTRD_SB2ST "U".
// D3 is the matrix of eigenvalues computed when Z is not computed
// and from the S resulting of DSYTRD_SB2ST "L".
//
// When ZCHKHBSTG is called, a number of matrix "sizes" ("n's"), a number
// of bandwidths ("k's"), and a number of matrix "types" are
// specified.  For each size ("n"), each bandwidth ("k") less than or
// equal to "n", and each _type of matrix, one matrix will be generated
// and used to test the hermitian banded reduction routine.  For each
// matrix, a number of tests will be performed:
//
// (1)     | A - V S V* | / ( |A| n ulp )  computed by ZHBTRD with
//                                         UPLO='U'
//
// (2)     | I - UU* | / ( n ulp )
//
// (3)     | A - V S V* | / ( |A| n ulp )  computed by ZHBTRD with
//                                         UPLO='L'
//
// (4)     | I - UU* | / ( n ulp )
//
// (5)     | D1 - D2 | / ( |D1| ulp )      where D1 is computed by
//                                         DSBTRD with UPLO='U' and
//                                         D2 is computed by
//                                         ZHETRD_HB2ST with UPLO='U'
//
// (6)     | D1 - D3 | / ( |D1| ulp )      where D1 is computed by
//                                         DSBTRD with UPLO='U' and
//                                         D3 is computed by
//                                         ZHETRD_HB2ST with UPLO='L'
//
// The "sizes" are specified by an array NN(1:NSIZES); the value of
// each element NN(j) specifies one size.
// The "types" are specified by a logical array DOTYPE( 1:NTYPES );
// if DOTYPE(j) is .TRUE., then matrix _type "j" will be generated.
// Currently, the list of possible types is:
//
// (1)  The zero matrix.
// (2)  The identity matrix.
//
// (3)  A diagonal matrix with evenly spaced entries
//      1, ..., ULP  and random signs.
//      (ULP = (first number larger than 1) - 1 )
// (4)  A diagonal matrix with geometrically spaced entries
//      1, ..., ULP  and random signs.
// (5)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//      and random signs.
//
// (6)  Same as (4), but multiplied by SQRT( overflow threshold )
// (7)  Same as (4), but multiplied by SQRT( underflow threshold )
//
// (8)  A matrix of the form  U* D U, where U is unitary and
//      D has evenly spaced entries 1, ..., ULP with random signs
//      on the diagonal.
//
// (9)  A matrix of the form  U* D U, where U is unitary and
//      D has geometrically spaced entries 1, ..., ULP with random
//      signs on the diagonal.
//
// (10) A matrix of the form  U* D U, where U is unitary and
//      D has "clustered" entries 1, ULP,..., ULP with random
//      signs on the diagonal.
//
// (11) Same as (8), but multiplied by SQRT( overflow threshold )
// (12) Same as (8), but multiplied by SQRT( underflow threshold )
//
// (13) Hermitian matrix with random entries chosen from (-1,1).
// (14) Same as (13), but multiplied by SQRT( overflow threshold )
// (15) Same as (13), but multiplied by SQRT( underflow threshold )
func Zchkhb2stg(nsizes *int, nn *[]int, nwdths *int, kk *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.CMatrix, lda *int, sd, se, d1, d2, d3 *mat.Vector, u *mat.CMatrix, ldu *int, work *mat.CVector, lwork *int, rwork, result *mat.Vector, info *int, t *testing.T) {
	var badnn, badnnb bool
	var cone, czero complex128
	var aninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, temp4, ten, two, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, itype, j, jc, jcol, jr, jsize, jtype, jwidth, k, kmax, lh, lw, maxtyp, mtypes, n, nerrs, nmats, nmax, ntest, ntestt int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kmagn := make([]int, 15)
	kmode := make([]int, 15)
	ktype := make([]int, 15)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	two = 2.0
	ten = 10.0
	half = one / two
	maxtyp = 15

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14] = 1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0

	//     Check for errors
	ntestt = 0
	(*info) = 0

	//     Important constants
	badnn = false
	nmax = 1
	for j = 1; j <= (*nsizes); j++ {
		nmax = maxint(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
	}

	badnnb = false
	kmax = 0
	for j = 1; j <= (*nsizes); j++ {
		kmax = maxint(kmax, (*kk)[j-1])
		if (*kk)[j-1] < 0 {
			badnnb = true
		}
	}
	kmax = minint(nmax-1, kmax)

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badnn {
		(*info) = -2
	} else if (*nwdths) < 0 {
		(*info) = -3
	} else if badnnb {
		(*info) = -4
	} else if (*ntypes) < 0 {
		(*info) = -5
	} else if (*lda) < kmax+1 {
		(*info) = -11
	} else if (*ldu) < nmax {
		(*info) = -15
	} else if (maxint(*lda, nmax)+1)*nmax > (*lwork) {
		(*info) = -17
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZCHKHBSTG"), -(*info))
		return
	}

	//     Quick return if possible
	if (*nsizes) == 0 || (*ntypes) == 0 || (*nwdths) == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)

	//     Loop over sizes, types
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		n = (*nn)[jsize-1]
		aninv = one / float64(maxint(1, n))

		for jwidth = 1; jwidth <= (*nwdths); jwidth++ {
			k = (*kk)[jwidth-1]
			if k > n {
				goto label180
			}
			k = maxint(0, minint(n-1, k))

			if (*nsizes) != 1 {
				mtypes = minint(maxtyp, *ntypes)
			} else {
				mtypes = minint(maxtyp+1, *ntypes)
			}

			for jtype = 1; jtype <= mtypes; jtype++ {
				if !(*dotype)[jtype-1] {
					goto label170
				}
				nmats = nmats + 1
				ntest = 0

				for j = 1; j <= 4; j++ {
					ioldsd[j-1] = (*iseed)[j-1]
				}

				//              Compute "A".
				//              Store as "Upper"; later, we will copy to other format.
				//
				//              Control parameters:
				//
				//                  KMAGN  KMODE        KTYPE
				//              =1  O(1)   clustered 1  zero
				//              =2  large  clustered 2  identity
				//              =3  small  exponential  (none)
				//              =4         arithmetic   diagonal, (w/ eigenvalues)
				//              =5         random log   hermitian, w/ eigenvalues
				//              =6         random       (none)
				//              =7                      random diagonal
				//              =8                      random hermitian
				//              =9                      positive definite
				//              =10                     diagonally dominant tridiagonal
				if mtypes > maxtyp {
					goto label100
				}

				itype = ktype[jtype-1]
				imode = kmode[jtype-1]

				//              Compute norm
				switch kmagn[jtype-1] {
				case 1:
					goto label40
				case 2:
					goto label50
				case 3:
					goto label60
				}

			label40:
				;
				anorm = one
				goto label70

			label50:
				;
				anorm = (rtovfl * ulp) * aninv
				goto label70

			label60:
				;
				anorm = rtunfl * float64(n) * ulpinv
				goto label70

			label70:
				;

				golapack.Zlaset('F', lda, &n, &czero, &czero, a, lda)
				iinfo = 0
				if jtype <= 15 {
					cond = ulpinv
				} else {
					cond = ulpinv * aninv / ten
				}

				//              Special Matrices -- Identity & Jordan block
				//
				//                 Zero
				if itype == 1 {
					iinfo = 0

				} else if itype == 2 {
					//                 Identity
					for jcol = 1; jcol <= n; jcol++ {
						a.SetRe(k+1-1, jcol-1, anorm)
					}

				} else if itype == 4 {
					//                 Diagonal Matrix, [Eigen]values Specified
					matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'Q', a.Off(k+1-1, 0), lda, work, &iinfo)

				} else if itype == 5 {
					//                 Hermitian, eigenvalues specified
					matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, &k, &k, 'Q', a, lda, work, &iinfo)

				} else if itype == 7 {
					//                 Diagonal, random eigenvalues
					matgen.Zlatmr(&n, &n, 'S', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'Q', a.Off(k+1-1, 0), lda, &idumma, &iinfo)

				} else if itype == 8 {
					//                 Hermitian, random eigenvalues
					matgen.Zlatmr(&n, &n, 'S', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &k, &k, &zero, &anorm, 'Q', a, lda, &idumma, &iinfo)

				} else if itype == 9 {
					//                 Positive definite, eigenvalues specified.
					matgen.Zlatms(&n, &n, 'S', iseed, 'P', rwork, &imode, &cond, &anorm, &k, &k, 'Q', a, lda, work.Off(n+1-1), &iinfo)

				} else if itype == 10 {
					//                 Positive definite tridiagonal, eigenvalues specified.
					if n > 1 {
						k = maxint(1, k)
					}
					matgen.Zlatms(&n, &n, 'S', iseed, 'P', rwork, &imode, &cond, &anorm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), 'Q', a.Off(k-1, 0), lda, work, &iinfo)
					for i = 2; i <= n; i++ {
						temp1 = a.GetMag(k-1, i-1) / math.Sqrt(cmplx.Abs(a.Get(k+1-1, i-1-1)*a.Get(k+1-1, i-1)))
						if temp1 > half {
							a.SetRe(k-1, i-1, half*math.Sqrt(cmplx.Abs(a.Get(k+1-1, i-1-1)*a.Get(k+1-1, i-1))))
						}
					}

				} else {

					iinfo = 1
				}

				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKHBSTG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					return
				}

			label100:
				;

				//              Call ZHBTRD to compute S and U from upper triangle.
				golapack.Zlacpy(' ', toPtr(k+1), &n, a, lda, work.CMatrix(*lda, opts), lda)

				ntest = 1
				golapack.Zhbtrd('V', 'U', &n, &k, work.CMatrix(*lda, opts), lda, sd, se, u, ldu, work.Off((*lda)*n+1-1), &iinfo)

				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKHBSTG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHBTRD(U)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(0, ulpinv)
						goto label150
					}
				}

				//              Do tests 1 and 2
				Zhbt21('U', &n, &k, func() *int { y := 1; return &y }(), a, lda, sd, se, u, ldu, work, rwork, result.Off(0))

				//              Before converting A into lower for DSBTRD, run DSYTRD_SB2ST
				//              otherwise matrix A will be converted to lower and then need
				//              to be converted back to upper in order to run the upper case
				//              ofDSYTRD_SB2ST
				//
				//              Compute D1 the eigenvalues resulting from the tridiagonal
				//              form using the DSBTRD and used as reference to compare
				//              with the DSYTRD_SB2ST routine
				//
				//              Compute D1 from the DSBTRD and used as reference for the
				//              DSYTRD_SB2ST
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d1, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}

				golapack.Zsteqr('N', &n, d1, rwork, work.CMatrix(*ldu, opts), ldu, rwork.Off(n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKHBSTG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(N)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(4, ulpinv)
						goto label150
					}
				}

				//              DSYTRD_SB2ST Upper case is used to compute D2.
				//              Note to set SD and SE to zero to be sure not reusing
				//              the one from above. Compare it with D1 computed
				//              using the DSBTRD.
				golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, sd.Matrix(1, opts), func() *int { y := 1; return &y }())
				golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, se.Matrix(1, opts), func() *int { y := 1; return &y }())
				golapack.Zlacpy(' ', toPtr(k+1), &n, a, lda, u, ldu)
				lh = maxint(1, 4*n)
				lw = (*lwork) - lh
				golapack.Zhetrdhb2st('N', 'N', 'U', &n, &k, u, ldu, sd, se, work, &lh, work.Off(lh+1-1), &lw, &iinfo)

				//              Compute D2 from the DSYTRD_SB2ST Upper case
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d2, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}

				golapack.Zsteqr('N', &n, d2, rwork, work.CMatrix(*ldu, opts), ldu, rwork.Off(n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKHBSTG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(N)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(4, ulpinv)
						goto label150
					}
				}

				//              Convert A from Upper-Triangle-Only storage to
				//              Lower-Triangle-Only storage.
				for jc = 1; jc <= n; jc++ {
					for jr = 0; jr <= minint(k, n-jc); jr++ {
						a.Set(jr+1-1, jc-1, a.GetConj(k+1-jr-1, jc+jr-1))
					}
				}
				for jc = n + 1 - k; jc <= n; jc++ {
					for jr = minint(k, n-jc) + 1; jr <= k; jr++ {
						a.SetRe(jr+1-1, jc-1, zero)
					}
				}

				//              Call ZHBTRD to compute S and U from lower triangle
				golapack.Zlacpy(' ', toPtr(k+1), &n, a, lda, work.CMatrix(*lda, opts), lda)

				ntest = 3
				golapack.Zhbtrd('V', 'L', &n, &k, work.CMatrix(*lda, opts), lda, sd, se, u, ldu, work.Off((*lda)*n+1-1), &iinfo)

				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKHBSTG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHBTRD(L)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(2, ulpinv)
						goto label150
					}
				}
				ntest = 4

				//              Do tests 3 and 4
				Zhbt21('L', &n, &k, func() *int { y := 1; return &y }(), a, lda, sd, se, u, ldu, work, rwork, result.Off(2))

				//              DSYTRD_SB2ST Lower case is used to compute D3.
				//              Note to set SD and SE to zero to be sure not reusing
				//              the one from above. Compare it with D1 computed
				//              using the DSBTRD.
				golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, sd.Matrix(1, opts), func() *int { y := 1; return &y }())
				golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, se.Matrix(1, opts), func() *int { y := 1; return &y }())
				golapack.Zlacpy(' ', toPtr(k+1), &n, a, lda, u, ldu)
				lh = maxint(1, 4*n)
				lw = (*lwork) - lh
				golapack.Zhetrdhb2st('N', 'N', 'L', &n, &k, u, ldu, sd, se, work, &lh, work.Off(lh+1-1), &lw, &iinfo)

				//              Compute D3 from the 2-stage Upper case
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d3, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}

				golapack.Zsteqr('N', &n, d3, rwork, work.CMatrix(*ldu, opts), ldu, rwork.Off(n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKHBSTG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(N)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(5, ulpinv)
						goto label150
					}
				}

				//              Do Tests 3 and 4 which are similar to 11 and 12 but with the
				//              D1 computed using the standard 1-stage reduction as reference
				ntest = 6
				temp1 = zero
				temp2 = zero
				temp3 = zero
				temp4 = zero

				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, d1.GetMag(j-1), d2.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					temp3 = maxf64(temp3, d1.GetMag(j-1), d3.GetMag(j-1))
					temp4 = maxf64(temp4, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}

				result.Set(4, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))
				result.Set(5, temp4/maxf64(unfl, ulp*maxf64(temp3, temp4)))

				//              End of Loop -- Check for RESULT(j) > THRESH
			label150:
				;
				ntestt = ntestt + ntest

				//              Print out tests which fail.
				for jr = 1; jr <= ntest; jr++ {
					if result.Get(jr-1) >= (*thresh) {
						t.Fail()
						//                    If this is the first test to fail,
						//                    print a header to the data file.
						if nerrs == 0 {
							fmt.Printf("\n %3s -- Complex Hermitian Banded Tridiagonal Reduction Routines\n", "ZHB")
							fmt.Printf(" Matrix types (see DCHK23 for details): \n")
							fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
							fmt.Printf(" Dense %s Banded Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Hermitian")
							fmt.Printf("\n Tests performed:   (S is Tridiag,  U is %s,\n                    %s means %s.\n UPLO='U':\n  1= | A - U S U%c | / ( |A| n ulp )       2= | I - U U%c | / ( n ulp )\n UPLO='L':\n  3= | A - U S U%c | / ( |A| n ulp )       4= | I - U U%c | / ( n ulp )\n Eig check:\n  5= | D1 - D2 | / ( |D1| ulp )           6= | D1 - D3 | / ( |D1| ulp )          \n", "unitary", "*", "conjugate transpose", '*', '*', '*', '*')
						}
						nerrs = nerrs + 1
						fmt.Printf(" N=%5d, K=%4d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", n, k, ioldsd, jtype, jr, result.Get(jr-1))
					}
				}

			label170:
			}
		label180:
		}
	}

	//     Summary
	Dlasum([]byte("ZHB"), &nerrs, &ntestt)
}
