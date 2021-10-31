package eig

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchksbstg tests the reduction of a symmetric band matrix to tridiagonal
// form, used with the symmetric eigenvalue problem.
//
// Dsbtrd factors a symmetric band matrix A as  U S U' , where ' means
// transpose, S is symmetric tridiagonal, and U is orthogonal.
// Dsbtrd can use either just the lower or just the upper triangle
// of A; dchksbstg checks both cases.
//
// DSYTRD_SB2ST factors a symmetric band matrix A as  U S U' ,
// where ' means transpose, S is symmetric tridiagonal, and U is
// orthogonal. DSYTRD_SB2ST can use either just the lower or just
// the upper triangle of A; dchksbstg checks both cases.
//
// Dsteqr factors S as  Z D1 Z'.
// D1 is the matrix of eigenvalues computed when Z is not computed
// and from the S resulting of Dsbtrd "U" (used as reference for DSYTRD_SB2ST)
// D2 is the matrix of eigenvalues computed when Z is not computed
// and from the S resulting of DSYTRD_SB2ST "U".
// D3 is the matrix of eigenvalues computed when Z is not computed
// and from the S resulting of DSYTRD_SB2ST "L".
//
// When dchksbstg is called, a number of matrix "sizes" ("n's"), a number
// of bandwidths ("k's"), and a number of matrix "types" are
// specified.  For each size ("n"), each bandwidth ("k") less than or
// equal to "n", and each type of matrix, one matrix will be generated
// and used to test the symmetric banded reduction routine.  For each
// matrix, a number of tests will be performed:
//
// (1)     | A - V S V' | / ( |A| n ulp )  computed by Dsbtrd with
//                                         UPLO='U'
//
// (2)     | I - UU' | / ( n ulp )
//
// (3)     | A - V S V' | / ( |A| n ulp )  computed by Dsbtrd with
//                                         UPLO='L'
//
// (4)     | I - UU' | / ( n ulp )
//
// (5)     | D1 - D2 | / ( |D1| ulp )      where D1 is computed by
//                                         Dsbtrd with UPLO='U' and
//                                         D2 is computed by
//                                         DSYTRD_SB2ST with UPLO='U'
//
// (6)     | D1 - D3 | / ( |D1| ulp )      where D1 is computed by
//                                         Dsbtrd with UPLO='U' and
//                                         D3 is computed by
//                                         DSYTRD_SB2ST with UPLO='L'
//
// The "sizes" are specified by an array NN(1:NSIZES); the value of
// each element NN(j) specifies one size.
// The "types" are specified by a logical array DOTYPE( 1:NTYPES );
// if DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
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
// (8)  A matrix of the form  U' D U, where U is orthogonal and
//      D has evenly spaced entries 1, ..., ULP with random signs
//      on the diagonal.
//
// (9)  A matrix of the form  U' D U, where U is orthogonal and
//      D has geometrically spaced entries 1, ..., ULP with random
//      signs on the diagonal.
//
// (10) A matrix of the form  U' D U, where U is orthogonal and
//      D has "clustered" entries 1, ULP,..., ULP with random
//      signs on the diagonal.
//
// (11) Same as (8), but multiplied by SQRT( overflow threshold )
// (12) Same as (8), but multiplied by SQRT( underflow threshold )
//
// (13) Symmetric matrix with random entries chosen from (-1,1).
// (14) Same as (13), but multiplied by SQRT( overflow threshold )
// (15) Same as (13), but multiplied by SQRT( underflow threshold )
func dchksb2stg(nsizes int, nn []int, nwdths int, kk []int, ntypes int, dotype []bool, iseed []int, thresh float64, nounit int, a *mat.Matrix, sd, se, d1, d2, d3 *mat.Vector, u *mat.Matrix, work *mat.Vector, lwork int, result *mat.Vector, t *testing.T) (nerrs, ntestt int, err error) {
	var badnn, badnnb bool
	var aninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, temp4, ten, two, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, itype, j, jc, jcol, jr, jsize, jtype, jwidth, k, kmax, lh, lw, maxtyp, mtypes, n, nmats, nmax, ntest int

	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kmagn := []int{1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3}
	kmode := []int{0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0}
	ktype := []int{1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8}

	zero = 0.0
	one = 1.0
	two = 2.0
	ten = 10.0
	half = one / two
	maxtyp = 15

	//     Check for errors
	ntestt = 0

	//     Important constants
	badnn = false
	nmax = 1
	for j = 1; j <= nsizes; j++ {
		nmax = max(nmax, nn[j-1])
		if nn[j-1] < 0 {
			badnn = true
		}
	}

	badnnb = false
	kmax = 0
	for j = 1; j <= nsizes; j++ {
		kmax = max(kmax, kk[j-1])
		if kk[j-1] < 0 {
			badnnb = true
		}
	}
	kmax = min(nmax-1, kmax)

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsize=%v", nsizes)
	} else if badnn {
		err = fmt.Errorf("badnn: nn=%v", nn)
	} else if nwdths < 0 {
		err = fmt.Errorf("nwdths < 0: nwdths=%v", nwdths)
	} else if badnnb {
		err = fmt.Errorf("badnnb: kk=%v", kk)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if a.Rows < kmax+1 {
		err = fmt.Errorf("a.Rows < kmax+1: a.Rows=%v, kmax=%v", a.Rows, kmax)
	} else if u.Rows < nmax {
		err = fmt.Errorf("u.Rows < nmax: u.Rows=%v, nmax=%v", u.Rows, nmax)
	} else if (max(a.Rows, nmax)+1)*nmax > lwork {
		err = fmt.Errorf("(max(a.Rows, nmax)+1)*nmax > lwork: a.Rows=%v, nmax=%v, lwork=%v", a.Rows, nmax, lwork)
	}

	if err != nil {
		gltest.Xerbla2("dchksbstg", err)
		return
	}

	//     Quick return if possible
	if nsizes == 0 || ntypes == 0 || nwdths == 0 {
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

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		aninv = one / float64(max(1, n))

		for jwidth = 1; jwidth <= nwdths; jwidth++ {
			k = kk[jwidth-1]
			if k > n {
				goto label180
			}
			k = max(0, min(n-1, k))

			if nsizes != 1 {
				mtypes = min(maxtyp, ntypes)
			} else {
				mtypes = min(maxtyp+1, ntypes)
			}

			for jtype = 1; jtype <= mtypes; jtype++ {
				if !dotype[jtype-1] {
					goto label170
				}
				nmats = nmats + 1
				ntest = 0

				for j = 1; j <= 4; j++ {
					ioldsd[j-1] = iseed[j-1]
				}

				//              Compute "A".
				//              Store as 'U'; later, we will copy to other format.
				//
				//              Control parameters:
				//
				//                  KMAGN  KMODE        KTYPE
				//              =1  O(1)   clustered 1  zero
				//              =2  large  clustered 2  identity
				//              =3  small  exponential  (none)
				//              =4         arithmetic   diagonal, (w/ eigenvalues)
				//              =5         random log   symmetric, w/ eigenvalues
				//              =6         random       (none)
				//              =7                      random diagonal
				//              =8                      random symmetric
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
				golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
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
						a.Set(k, jcol-1, anorm)
					}

				} else if itype == 4 {
					//                 Diagonal Matrix, [Eigen]values Specified
					iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, 0, 0, 'Q', a.Off(k, 0), work.Off(n))

				} else if itype == 5 {
					//                 Symmetric, eigenvalues specified
					iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, k, k, 'Q', a, work.Off(n))

				} else if itype == 7 {
					//                 Diagonal, random eigenvalues
					iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'Q', a.Off(k, 0), &idumma)

				} else if itype == 8 {
					//                 Symmetric, random eigenvalues
					iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, k, k, zero, anorm, 'Q', a, &idumma)
					//
				} else if itype == 9 {
					//                 Positive definite, eigenvalues specified.
					iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'P', work, imode, cond, anorm, k, k, 'Q', a, work.Off(n))

				} else if itype == 10 {
					//                 Positive definite tridiagonal, eigenvalues specified.
					if n > 1 {
						k = max(1, k)
					}
					iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'P', work, imode, cond, anorm, 1, 1, 'Q', a.Off(k-1, 0), work.Off(n))
					for i = 2; i <= n; i++ {
						temp1 = math.Abs(a.Get(k-1, i-1)) / math.Sqrt(math.Abs(a.Get(k, i-1-1)*a.Get(k, i-1)))
						if temp1 > half {
							a.Set(k-1, i-1, half*math.Sqrt(math.Abs(a.Get(k, i-1-1)*a.Get(k, i-1))))
						}
					}

				} else {

					iinfo = 1
				}

				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" dchksbstg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

			label100:
				;

				//              Call Dsbtrd to compute S and U from upper triangle.
				golapack.Dlacpy(Full, k+1, n, a, work.Matrix(a.Rows, opts))

				ntest = 1
				if err = golapack.Dsbtrd('V', Upper, n, k, work.Matrix(a.Rows, opts), sd, se, u, work.Off(a.Rows*n)); err != nil {
					t.Fail()
					fmt.Printf(" dchksbstg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbtrd(U)", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if iinfo < 0 {
						return
					} else {
						result.Set(0, ulpinv)
						goto label150
					}
				}

				//              Do tests 1 and 2
				dsbt21(Upper, n, k, 1, a, sd, se, u, work, result)

				//              Before converting A into lower for Dsbtrd, run DSYTRD_SB2ST
				//              otherwise matrix A will be converted to lower and then need
				//              to be converted back to upper in order to run the upper case
				//              ofDSYTRD_SB2ST
				//
				//              Compute D1 the eigenvalues resulting from the tridiagonal
				//              form using the Dsbtrd and used as reference to compare
				//              with the DSYTRD_SB2ST routine
				//
				//              Compute D1 from the Dsbtrd and used as reference for the
				//              DSYTRD_SB2ST
				goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				if iinfo, err = golapack.Dsteqr('N', n, d1, work, work.MatrixOff(n, u.Rows, opts), work.Off(n)); err != nil || iinfo != 0 {
					t.Fail()
					fmt.Printf(" dchksbstg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(N)", iinfo, n, jtype, ioldsd)
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
				//              using the Dsbtrd.
				golapack.Dlaset(Full, n, 1, zero, zero, sd.Matrix(n, opts))
				golapack.Dlaset(Full, n, 1, zero, zero, se.Matrix(n, opts))
				golapack.Dlacpy(Full, k+1, n, a, u)
				lh = max(1, 4*n)
				lw = lwork - lh
				if err = golapack.DsytrdSb2st('N', 'N', Upper, n, k, u, sd, se, work, lh, work.Off(lh), lw); err != nil {
					panic(err)
				}

				//              Compute D2 from the DSYTRD_SB2ST Upper case
				goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				if iinfo, err = golapack.Dsteqr('N', n, d2, work, work.MatrixOff(n, u.Rows, opts), work.Off(n)); err != nil || iinfo != 0 {
					t.Fail()
					fmt.Printf(" dchksbstg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(N)", iinfo, n, jtype, ioldsd)
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
					for jr = 0; jr <= min(k, n-jc); jr++ {
						a.Set(jr, jc-1, a.Get(k+1-jr-1, jc+jr-1))
					}
				}
				for jc = n + 1 - k; jc <= n; jc++ {
					for jr = min(k, n-jc) + 1; jr <= k; jr++ {
						a.Set(jr, jc-1, zero)
					}
				}

				//              Call Dsbtrd to compute S and U from lower triangle
				golapack.Dlacpy(Full, k+1, n, a, work.Matrix(a.Rows, opts))

				ntest = 3
				if err = golapack.Dsbtrd('V', Lower, n, k, work.Matrix(a.Rows, opts), sd, se, u, work.Off(a.Rows*n)); err != nil {
					t.Fail()
					fmt.Printf(" dchksbstg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbtrd(L)", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if iinfo < 0 {
						return
					} else {
						result.Set(2, ulpinv)
						goto label150
					}
				}
				ntest = 4

				//              Do tests 3 and 4
				dsbt21(Lower, n, k, 1, a, sd, se, u, work, result.Off(2))

				//              DSYTRD_SB2ST Lower case is used to compute D3.
				//              Note to set SD and SE to zero to be sure not reusing
				//              the one from above. Compare it with D1 computed
				//              using the Dsbtrd.
				golapack.Dlaset(Full, n, 1, zero, zero, sd.Matrix(n, opts))
				golapack.Dlaset(Full, n, 1, zero, zero, se.Matrix(n, opts))
				golapack.Dlacpy(Full, k+1, n, a, u)
				lh = max(1, 4*n)
				lw = lwork - lh
				if err = golapack.DsytrdSb2st('N', 'N', Lower, n, k, u, sd, se, work, lh, work.Off(lh), lw); err != nil {
					panic(err)
				}

				//              Compute D3 from the 2-stage Upper case
				goblas.Dcopy(n, sd.Off(0, 1), d3.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				if iinfo, err = golapack.Dsteqr('N', n, d3, work, work.MatrixOff(n, u.Rows, opts), work.Off(n)); err != nil || iinfo != 0 {
					t.Fail()
					fmt.Printf(" dchksbstg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(N)", iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(5, ulpinv)
						goto label150
					}
				}

				//
				//              Do Tests 3 and 4 which are similar to 11 and 12 but with the
				//              D1 computed using the standard 1-stage reduction as reference
				ntest = 6
				temp1 = zero
				temp2 = zero
				temp3 = zero
				temp4 = zero

				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					temp3 = math.Max(temp3, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp4 = math.Max(temp4, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}

				result.Set(4, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
				result.Set(5, temp4/math.Max(unfl, ulp*math.Max(temp3, temp4)))

				//              End of Loop -- Check for RESULT(j) > THRESH
			label150:
				;
				ntestt = ntestt + ntest

				//              Print out tests which fail.
				for jr = 1; jr <= ntest; jr++ {
					if result.Get(jr-1) >= thresh {
						t.Fail()
						//                    If this is the first test to fail,
						//                    print a header to the data file.
						if nerrs == 0 {
							fmt.Printf("\n %3s -- Real Symmetric Banded Tridiagonal Reduction Routines\n", "DSB")
							fmt.Printf(" Matrix types (see dchksbstg for details): \n")
							fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
							fmt.Printf(" Dense %s Banded Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Symmetric")
							fmt.Printf("\n Tests performed:   (S is Tridiag,  U is %s,\n                    %s means %s.\n UPLO='U':\n  1= | A - U S U%c | / ( |A| n ulp )       2= | I - U U%c | / ( n ulp )\n UPLO='L':\n  3= | A - U S U%c | / ( |A| n ulp )       4= | I - U U%c | / ( n ulp )\n Eig check:\n  5= | D1 - D2 | / ( |D1| ulp )           6= | D1 - D3 | / ( |D1| ulp )          \n", "orthogonal", "'", "transpose", '\'', '\'', '\'', '\'')
						}
						nerrs = nerrs + 1
						fmt.Printf(" n=%5d, K=%4d, seed=%4d, type %2d, test(%2d)=%10.3f\n", n, k, ioldsd, jtype, jr, result.Get(jr-1))
					}
				}

			label170:
			}
		label180:
		}
	}

	//     Summary
	// dlasum("Dsb", nerrs, ntestt)

	return
}
