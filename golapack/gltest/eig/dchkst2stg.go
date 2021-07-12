package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dchkst2stg checks the symmetric eigenvalue problem routines
// using the 2-stage reduction techniques. Since the generation
// of Q or the vectors is not available in this release, we only
// compare the eigenvalue resulting when using the 2-stage to the
// one considered as reference using the standard 1-stage reduction
// DSYTRD. For that, we call the standard DSYTRD and compute D1 using
// DSTEQR, then we call the 2-stage DSYTRD_2STAGE with Upper and Lower
// and we compute D2 and D3 using DSTEQR and then we replaced tests
// 3 and 4 by tests 11 and 12. test 1 and 2 remain to verify that
// the 1-stage results are OK and can be trusted.
// This testing routine will converge to the DCHKST in the next
// release when vectors and generation of Q will be implemented.
//
//    DSYTRD factors A as  U S U' , where ' means transpose,
//    S is symmetric tridiagonal, and U is orthogonal.
//    DSYTRD can use either just the lower or just the upper triangle
//    of A; DCHKST2STG checks both cases.
//    U is represented as a product of Householder
//    transformations, whose vectors are stored in the first
//    n-1 columns of V, and whose scale factors are in TAU.
//
//    DSPTRD does the same as DSYTRD, except that A and V are stored
//    in "packed" format.
//
//    DORGTR constructs the matrix U from the contents of V and TAU.
//
//    DOPGTR constructs the matrix U from the contents of VP and TAU.
//
//    DSTEQR factors S as  Z D1 Z' , where Z is the orthogonal
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal.  D2 is the matrix of
//    eigenvalues computed when Z is not computed.
//
//    DSTERF computes D3, the matrix of eigenvalues, by the
//    PWK method, which does not yield eigenvectors.
//
//    DPTEQR factors S as  Z4 D4 Z4' , for a
//    symmetric positive definite tridiagonal matrix.
//    D5 is the matrix of eigenvalues computed when Z is not
//    computed.
//
//    DSTEBZ computes selected eigenvalues.  WA1, WA2, and
//    WA3 will denote eigenvalues computed to high
//    absolute accuracy, with different range options.
//    WR will denote eigenvalues computed to high relative
//    accuracy.
//
//    DSTEIN computes Y, the eigenvectors of S, given the
//    eigenvalues.
//
//    DSTEDC factors S as Z D1 Z' , where Z is the orthogonal
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option). It may also
//    update an input orthogonal matrix, usually the output
//    from DSYTRD/DORGTR or DSPTRD/DOPGTR ('V' option). It may
//    also just compute eigenvalues ('N' option).
//
//    DSTEMR factors S as Z D1 Z' , where Z is the orthogonal
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option).  DSTEMR
//    uses the Relatively Robust Representation whenever possible.
//
// When DCHKST2STG is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each type of matrix, one matrix will be generated and used
// to test the symmetric eigenroutines.  For each matrix, a number
// of tests will be performed:
//
// (1)     | A - V S V' | / ( |A| n ulp ) DSYTRD( UPLO='U', ... )
//
// (2)     | I - UV' | / ( n ulp )        DORGTR( UPLO='U', ... )
//
// (3)     | A - V S V' | / ( |A| n ulp ) DSYTRD( UPLO='L', ... )
//         replaced by | D1 - D2 | / ( |D1| ulp ) where D1 is the
//         eigenvalue matrix computed using S and D2 is the
//         eigenvalue matrix computed using S_2stage the output of
//         DSYTRD_2STAGE("N", "U",....). D1 and D2 are computed
//         via DSTEQR('N',...)
//
// (4)     | I - UV' | / ( n ulp )        DORGTR( UPLO='L', ... )
//         replaced by | D1 - D3 | / ( |D1| ulp ) where D1 is the
//         eigenvalue matrix computed using S and D3 is the
//         eigenvalue matrix computed using S_2stage the output of
//         DSYTRD_2STAGE("N", "L",....). D1 and D3 are computed
//         via DSTEQR('N',...)
//
// (5-8)   Same as 1-4, but for DSPTRD and DOPGTR.
//
// (9)     | S - Z D Z' | / ( |S| n ulp ) DSTEQR('V',...)
//
// (10)    | I - ZZ' | / ( n ulp )        DSTEQR('V',...)
//
// (11)    | D1 - D2 | / ( |D1| ulp )        DSTEQR('N',...)
//
// (12)    | D1 - D3 | / ( |D1| ulp )        DSTERF
//
// (13)    0 if the true eigenvalues (computed by sturm count)
//         of S are within THRESH of
//         those in D1.  2*THRESH if they are not.  (Tested using
//         DSTECH)
//
// For S positive definite,
//
// (14)    | S - Z4 D4 Z4' | / ( |S| n ulp ) DPTEQR('V',...)
//
// (15)    | I - Z4 Z4' | / ( n ulp )        DPTEQR('V',...)
//
// (16)    | D4 - D5 | / ( 100 |D4| ulp )       DPTEQR('N',...)
//
// When S is also diagonally dominant by the factor gamma < 1,
//
// (17)    math.Max | D4(i) - WR(i) | / ( |D4(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              DSTEBZ( 'A', 'E', ...)
//
// (18)    | WA1 - D3 | / ( |D3| ulp )          DSTEBZ( 'A', 'E', ...)
//
// (19)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//                                              DSTEBZ( 'I', 'E', ...)
//
// (20)    | S - Y WA1 Y' | / ( |S| n ulp )  DSTEBZ, SSTEIN
//
// (21)    | I - Y Y' | / ( n ulp )          DSTEBZ, SSTEIN
//
// (22)    | S - Z D Z' | / ( |S| n ulp )    DSTEDC('I')
//
// (23)    | I - ZZ' | / ( n ulp )           DSTEDC('I')
//
// (24)    | S - Z D Z' | / ( |S| n ulp )    DSTEDC('V')
//
// (25)    | I - ZZ' | / ( n ulp )           DSTEDC('V')
//
// (26)    | D1 - D2 | / ( |D1| ulp )           DSTEDC('V') and
//                                              DSTEDC('N')
//
// Test 27 is disabled at the moment because DSTEMR does not
// guarantee high relatvie accuracy.
//
// (27)    math.Max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              DSTEMR('V', 'A')
//
// (28)    math.Max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              DSTEMR('V', 'I')
//
// Tests 29 through 34 are disable at present because DSTEMR
// does not handle partial spectrum requests.
//
// (29)    | S - Z D Z' | / ( |S| n ulp )    DSTEMR('V', 'I')
//
// (30)    | I - ZZ' | / ( n ulp )           DSTEMR('V', 'I')
//
// (31)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         DSTEMR('N', 'I') vs. SSTEMR('V', 'I')
//
// (32)    | S - Z D Z' | / ( |S| n ulp )    DSTEMR('V', 'V')
//
// (33)    | I - ZZ' | / ( n ulp )           DSTEMR('V', 'V')
//
// (34)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         DSTEMR('N', 'V') vs. SSTEMR('V', 'V')
//
// (35)    | S - Z D Z' | / ( |S| n ulp )    DSTEMR('V', 'A')
//
// (36)    | I - ZZ' | / ( n ulp )           DSTEMR('V', 'A')
//
// (37)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         DSTEMR('N', 'A') vs. SSTEMR('V', 'A')
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
// (16) Same as (8), but diagonal elements are all positive.
// (17) Same as (9), but diagonal elements are all positive.
// (18) Same as (10), but diagonal elements are all positive.
// (19) Same as (16), but multiplied by SQRT( overflow threshold )
// (20) Same as (16), but multiplied by SQRT( underflow threshold )
// (21) A diagonally dominant tridiagonal matrix with geometrically
//      spaced diagonal entries 1, ..., ULP.
func Dchkst2stg(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.Matrix, lda *int, ap, sd, se, d1, d2, d3, d4, d5, wa1, wa2, wa3, wr *mat.Vector, u *mat.Matrix, ldu *int, v *mat.Matrix, vp, tau *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork *int, iwork *[]int, liwork *int, result *mat.Vector, info *int) {
	var badnn, srange, srel, tryrac bool
	var abstol, aninv, anorm, cond, eight, half, hun, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, temp4, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, il, imode, itemp, itype, iu, j, jc, jr, jsize, jtype, lgn, lh, liwedc, log2ui, lw, lwedc, m, m2, m3, maxtyp, mtypes, n, nap, nblock, nerrs, nmats, nmax, nsplit, ntest, ntestt int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	dumma := vf(1)

	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0
	ten = 10.0
	hun = 100.0
	half = one / two
	maxtyp = 21
	srange = false
	srel = false

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17], ktype[18], ktype[19], ktype[20] = 1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9, 9, 9, 10
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17], kmagn[18], kmagn[19], kmagn[20] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17], kmode[18], kmode[19], kmode[20] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 3, 1, 4, 4, 3

	//     Keep ftnchek happy
	idumma[0] = 1

	//     Check for errors
	ntestt = 0
	(*info) = 0

	//     Important constants
	badnn = false
	tryrac = true
	nmax = 1
	for j = 1; j <= (*nsizes); j++ {
		nmax = max(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
	}

	nblock = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRD"), []byte("L"), &nmax, toPtr(-1), toPtr(-1), toPtr(-1))
	nblock = min(nmax, max(1, nblock))

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badnn {
		(*info) = -2
	} else if (*ntypes) < 0 {
		(*info) = -3
	} else if (*lda) < nmax {
		(*info) = -9
	} else if (*ldu) < nmax {
		(*info) = -23
	} else if 2*int(math.Pow(float64(max(2, nmax)), 2)) > (*lwork) {
		(*info) = -29
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DCHKST2STG"), -(*info))
		return
	}

	//     Quick return if possible
	if (*nsizes) == 0 || (*ntypes) == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	golapack.Dlabad(&unfl, &ovfl)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	log2ui = int(math.Log(ulpinv) / math.Log(two))
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)

	//     Loop over sizes, types
	for i = 1; i <= 4; i++ {
		iseed2[i-1] = (*iseed)[i-1]
	}
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		n = (*nn)[jsize-1]
		if n > 0 {
			lgn = int(math.Log(float64(n)) / math.Log(two))
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			lwedc = 1 + 4*n + 2*n*lgn + 4*int(math.Pow(float64(n), 2))
			liwedc = 6 + 6*n + 5*n*lgn
		} else {
			lwedc = 8
			liwedc = 12
		}
		nap = (n * (n + 1)) / 2
		aninv = one / float64(max(1, n))

		if (*nsizes) != 1 {
			mtypes = min(maxtyp, *ntypes)
		} else {
			mtypes = min(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label300
			}
			nmats = nmats + 1
			ntest = 0

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			//           Compute "A"
			//
			//           Control parameters:
			//
			//               KMAGN  KMODE        KTYPE
			//           =1  O(1)   clustered 1  zero
			//           =2  large  clustered 2  identity
			//           =3  small  exponential  (none)
			//           =4         arithmetic   diagonal, (w/ eigenvalues)
			//           =5         random log   symmetric, w/ eigenvalues
			//           =6         random       (none)
			//           =7                      random diagonal
			//           =8                      random symmetric
			//           =9                      positive definite
			//           =10                     diagonally dominant tridiagonal
			if mtypes > maxtyp {
				goto label100
			}

			itype = ktype[jtype-1]
			imode = kmode[jtype-1]

			//           Compute norm
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
			golapack.Dlaset('F', lda, &n, &zero, &zero, a, lda)
			iinfo = 0
			if jtype <= 15 {
				cond = ulpinv
			} else {
				cond = ulpinv * aninv / ten
			}

			//           Special Matrices -- Identity & Jordan block
			//
			//              Zero
			if itype == 1 {
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jc = 1; jc <= n; jc++ {
					a.Set(jc-1, jc-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work.Off(n), &iinfo)

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work.Off(n), &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n), func() *int { y := 1; return &y }(), &one, work.Off(2*n), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n), func() *int { y := 1; return &y }(), &one, work.Off(2*n), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              Positive definite, eigenvalues specified.
				matgen.Dlatms(&n, &n, 'S', iseed, 'P', work, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work.Off(n), &iinfo)

			} else if itype == 10 {
				//              Positive definite tridiagonal, eigenvalues specified.
				matgen.Dlatms(&n, &n, 'S', iseed, 'P', work, &imode, &cond, &anorm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), 'N', a, lda, work.Off(n), &iinfo)
				for i = 2; i <= n; i++ {
					temp1 = math.Abs(a.Get(i-1-1, i-1)) / math.Sqrt(math.Abs(a.Get(i-1-1, i-1-1)*a.Get(i-1, i-1)))
					if temp1 > half {
						a.Set(i-1-1, i-1, half*math.Sqrt(math.Abs(a.Get(i-1-1, i-1-1)*a.Get(i-1, i-1))))
						a.Set(i-1, i-1-1, a.Get(i-1-1, i-1))
					}
				}

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				return
			}

		label100:
			;

			//           Call DSYTRD and DORGTR to compute S and U from
			//           upper triangle.
			golapack.Dlacpy('U', &n, &n, a, lda, v, ldu)

			ntest = 1
			golapack.Dsytrd('U', &n, v, ldu, sd, se, tau, work, lwork, &iinfo)

			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYTRD(U)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(0, ulpinv)
					goto label280
				}
			}

			golapack.Dlacpy('U', &n, &n, v, ldu, u, ldu)

			ntest = 2
			golapack.Dorgtr('U', &n, u, ldu, tau, work, lwork, &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DORGTR(U)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(1, ulpinv)
					goto label280
				}
			}

			//           Do tests 1 and 2
			Dsyt21(func() *int { y := 2; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), a, lda, sd, se, u, ldu, v, ldu, tau, work, result.Off(0))
			Dsyt21(func() *int { y := 3; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), a, lda, sd, se, u, ldu, v, ldu, tau, work, result.Off(1))

			//           Compute D1 the eigenvalues resulting from the tridiagonal
			//           form using the standard 1-stage algorithm and use it as a
			//           reference to compare with the 2-stage technique
			//
			//           Compute D1 from the 1-stage and used as reference for the
			//           2-stage
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			golapack.Dsteqr('N', &n, d1, work, work.MatrixOff(n, *ldu, opts), ldu, work.Off(n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(2, ulpinv)
					goto label280
				}
			}

			//           2-STAGE TRD Upper case is used to compute D2.
			//           Note to set SD and SE to zero to be sure not reusing
			//           the one from above. Compare it with D1 computed
			//           using the 1-stage.
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, sd.Matrix(n, opts), func() *int { y := 1; return &y }())
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, se.Matrix(n, opts), func() *int { y := 1; return &y }())
			golapack.Dlacpy('U', &n, &n, a, lda, v, ldu)
			lh = max(1, 4*n)
			lw = (*lwork) - lh
			golapack.Dsytrd2stage('N', 'U', &n, v, ldu, sd, se, tau, work, &lh, work.Off(lh), &lw, &iinfo)

			//           Compute D2 from the 2-stage Upper case
			goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			golapack.Dsteqr('N', &n, d2, work, work.MatrixOff(n, *ldu, opts), ldu, work.Off(n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(2, ulpinv)
					goto label280
				}
			}

			//           2-STAGE TRD Lower case is used to compute D3.
			//           Note to set SD and SE to zero to be sure not reusing
			//           the one from above. Compare it with D1 computed
			//           using the 1-stage.
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, sd.Matrix(n, opts), func() *int { y := 1; return &y }())
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, se.Matrix(n, opts), func() *int { y := 1; return &y }())
			golapack.Dlacpy('L', &n, &n, a, lda, v, ldu)
			golapack.Dsytrd2stage('N', 'L', &n, v, ldu, sd, se, tau, work, &lh, work.Off(lh), &lw, &iinfo)

			//           Compute D3 from the 2-stage Upper case
			goblas.Dcopy(n, sd.Off(0, 1), d3.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			golapack.Dsteqr('N', &n, d3, work, work.MatrixOff(n, *ldu, opts), ldu, work.Off(n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(3, ulpinv)
					goto label280
				}
			}

			//           Do Tests 3 and 4 which are similar to 11 and 12 but with the
			//           D1 computed using the standard 1-stage reduction as reference
			ntest = 4
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

			result.Set(2, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			result.Set(3, temp4/math.Max(unfl, ulp*math.Max(temp3, temp4)))

			//           Store the upper triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = 1; jr <= jc; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call DSPTRD and DOPGTR to compute S and U from AP
			goblas.Dcopy(nap, ap.Off(0, 1), vp.Off(0, 1))

			ntest = 5
			golapack.Dsptrd('U', &n, vp, sd, se, tau, &iinfo)

			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPTRD(U)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(4, ulpinv)
					goto label280
				}
			}

			ntest = 6
			golapack.Dopgtr('U', &n, vp, tau, u, ldu, work, &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DOPGTR(U)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(5, ulpinv)
					goto label280
				}
			}

			//           Do tests 5 and 6
			Dspt21(func() *int { y := 2; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, result.Off(4))
			Dspt21(func() *int { y := 3; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, result.Off(5))

			//           Store the lower triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = jc; jr <= n; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call DSPTRD and DOPGTR to compute S and U from AP
			goblas.Dcopy(nap, ap.Off(0, 1), vp.Off(0, 1))

			ntest = 7
			golapack.Dsptrd('L', &n, vp, sd, se, tau, &iinfo)

			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPTRD(L)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(6, ulpinv)
					goto label280
				}
			}

			ntest = 8
			golapack.Dopgtr('L', &n, vp, tau, u, ldu, work, &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DOPGTR(L)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(7, ulpinv)
					goto label280
				}
			}

			Dspt21(func() *int { y := 2; return &y }(), 'L', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, result.Off(6))
			Dspt21(func() *int { y := 3; return &y }(), 'L', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, result.Off(7))

			//           Call DSTEQR to compute D1, D2, and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset('F', &n, &n, &zero, &one, z, ldu)

			ntest = 9
			golapack.Dsteqr('V', &n, d1, work, z, ldu, work.Off(n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEQR(V)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(8, ulpinv)
					goto label280
				}
			}

			//           Compute D2
			goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			ntest = 11
			golapack.Dsteqr('N', &n, d2, work, work.MatrixOff(n, *ldu, opts), ldu, work.Off(n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(10, ulpinv)
					goto label280
				}
			}

			//           Compute D3 (using PWK method)
			goblas.Dcopy(n, sd.Off(0, 1), d3.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			ntest = 12
			golapack.Dsterf(&n, d3, work, &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTERF", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(11, ulpinv)
					goto label280
				}
			}

			//           Do Tests 9 and 10
			Dstt21(&n, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work, result.Off(8))

			//           Do Tests 11 and 12
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

			result.Set(10, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			result.Set(11, temp4/math.Max(unfl, ulp*math.Max(temp3, temp4)))

			//           Do Test 13 -- Sturm Sequence Test of Eigenvalues
			//                         Go up by factors of two until it succeeds
			ntest = 13
			temp1 = (*thresh) * (half - ulp)

			for j = 0; j <= log2ui; j++ {
				Dstech(&n, sd, se, d1, &temp1, work, &iinfo)
				if iinfo == 0 {
					goto label170
				}
				temp1 = temp1 * two
			}

		label170:
			;
			result.Set(12, temp1)

			//           For positive definite matrices ( JTYPE.GT.15 ) call DPTEQR
			//           and do tests 14, 15, and 16 .
			if jtype > 15 {
				//              Compute D4 and Z4
				goblas.Dcopy(n, sd.Off(0, 1), d4.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}
				golapack.Dlaset('F', &n, &n, &zero, &one, z, ldu)

				ntest = 14
				golapack.Dpteqr('V', &n, d4, work, z, ldu, work.Off(n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DPTEQR(V)", iinfo, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(13, ulpinv)
						goto label280
					}
				}

				//              Do Tests 14 and 15
				Dstt21(&n, func() *int { y := 0; return &y }(), sd, se, d4, dumma, z, ldu, work, result.Off(13))

				//              Compute D5
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				ntest = 16
				golapack.Dpteqr('N', &n, d5, work, z, ldu, work.Off(n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DPTEQR(N)", iinfo, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(15, ulpinv)
						goto label280
					}
				}

				//              Do Test 16
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Max(math.Abs(d4.Get(j-1)), math.Abs(d5.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d4.Get(j-1)-d5.Get(j-1)))
				}

				result.Set(15, temp2/math.Max(unfl, hun*ulp*math.Max(temp1, temp2)))
			} else {
				result.Set(13, zero)
				result.Set(14, zero)
				result.Set(15, zero)
			}

			//           Call DSTEBZ with different options and do tests 17-18.
			//
			//              If S is positive definite and diagonally dominant,
			//              ask for all eigenvalues with high relative accuracy.
			vl = zero
			vu = zero
			il = 0
			iu = 0
			if jtype == 21 {
				ntest = 17
				abstol = unfl + unfl
				golapack.Dstebz('A', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m, &nsplit, wr, iwork, toSlice(iwork, n), work, toSlice(iwork, 2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(A,rel)", iinfo, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(16, ulpinv)
						goto label280
					}
				}

				//              Do test 17
				temp2 = two * (two*float64(n) - one) * ulp * (one + eight*math.Pow(half, 2)) / math.Pow(one-half, 4)

				temp1 = zero
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j))/(abstol+math.Abs(d4.Get(j-1))))
				}

				result.Set(16, temp1/temp2)
			} else {
				result.Set(16, zero)
			}

			//           Now ask for all eigenvalues with high absolute accuracy.
			ntest = 18
			abstol = unfl + unfl
			golapack.Dstebz('A', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m, &nsplit, wa1, iwork, toSlice(iwork, n), work, toSlice(iwork, 2*n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(A)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(17, ulpinv)
					goto label280
				}
			}

			//           Do test 18
			temp1 = zero
			temp2 = zero
			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(math.Abs(d3.Get(j-1)), math.Abs(wa1.Get(j-1))))
				temp2 = math.Max(temp2, math.Abs(d3.Get(j-1)-wa1.Get(j-1)))
			}

			result.Set(17, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			//           Choose random values for IL and IU, and ask for the
			//           IL-th through IU-th eigenvalues.
			ntest = 19
			if n <= 1 {
				il = 1
				iu = n
			} else {
				il = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
				iu = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
				if iu < il {
					itemp = iu
					iu = il
					il = itemp
				}
			}

			golapack.Dstebz('I', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m2, &nsplit, wa2, iwork, toSlice(iwork, n), work, toSlice(iwork, 2*n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(I)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(18, ulpinv)
					goto label280
				}
			}

			//           Determine the values VL and VU of the IL-th and IU-th
			//           eigenvalues and ask for all eigenvalues in this range.
			if n > 0 {
				if il != 1 {
					vl = wa1.Get(il-1) - math.Max(half*(wa1.Get(il-1)-wa1.Get(il-1-1)), math.Max(ulp*anorm, two*rtunfl))
				} else {
					vl = wa1.Get(0) - math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ulp*anorm, two*rtunfl))
				}
				if iu != n {
					vu = wa1.Get(iu-1) + math.Max(half*(wa1.Get(iu)-wa1.Get(iu-1)), math.Max(ulp*anorm, two*rtunfl))
				} else {
					vu = wa1.Get(n-1) + math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ulp*anorm, two*rtunfl))
				}
			} else {
				vl = zero
				vu = one
			}

			golapack.Dstebz('V', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m3, &nsplit, wa3, iwork, toSlice(iwork, n), work, toSlice(iwork, 2*n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(V)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(18, ulpinv)
					goto label280
				}
			}

			if m3 == 0 && n != 0 {
				result.Set(18, ulpinv)
				goto label280
			}

			//           Do test 19
			temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
			temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
			if n > 0 {
				temp3 = math.Max(math.Abs(wa1.Get(n-1)), math.Abs(wa1.Get(0)))
			} else {
				temp3 = zero
			}

			result.Set(18, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			//           Call DSTEIN to compute eigenvectors corresponding to
			//           eigenvalues in WA1.  (First call DSTEBZ again, to make sure
			//           it returns these eigenvalues in the correct order.)
			ntest = 21
			golapack.Dstebz('A', 'B', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m, &nsplit, wa1, iwork, toSlice(iwork, n), work, toSlice(iwork, 2*n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(A,B)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			golapack.Dstein(&n, sd, se, &m, wa1, iwork, toSlice(iwork, n), z, ldu, work, toSlice(iwork, 2*n), toSlice(iwork, 3*n), &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEIN", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			//           Do tests 20 and 21
			Dstt21(&n, func() *int { y := 0; return &y }(), sd, se, wa1, dumma, z, ldu, work, result.Off(19))

			//           Call DSTEDC(I) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset('F', &n, &n, &zero, &one, z, ldu)

			ntest = 22
			golapack.Dstedc('I', &n, d1, work, z, ldu, work.Off(n), toPtr(lwedc-n), iwork, &liwedc, &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEDC(I)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(21, ulpinv)
					goto label280
				}
			}

			//           Do Tests 22 and 23
			Dstt21(&n, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work, result.Off(21))

			//           Call DSTEDC(V) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset('F', &n, &n, &zero, &one, z, ldu)

			ntest = 24
			golapack.Dstedc('V', &n, d1, work, z, ldu, work.Off(n), toPtr(lwedc-n), iwork, &liwedc, &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEDC(V)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(23, ulpinv)
					goto label280
				}
			}

			//           Do Tests 24 and 25
			Dstt21(&n, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work, result.Off(23))

			//           Call DSTEDC(N) to compute D2, do tests.
			//
			//           Compute D2
			goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset('F', &n, &n, &zero, &one, z, ldu)

			ntest = 26
			golapack.Dstedc('N', &n, d2, work, z, ldu, work.Off(n), toPtr(lwedc-n), iwork, &liwedc, &iinfo)
			if iinfo != 0 {
				fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEDC(N)", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(25, ulpinv)
					goto label280
				}
			}

			//           Do Test 26
			temp1 = zero
			temp2 = zero

			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
				temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
			}

			result.Set(25, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			//           Only test DSTEMR if IEEE compliant
			if Ilaenv(func() *int { y := 10; return &y }(), []byte("DSTEMR"), []byte("VA"), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }()) == 1 && Ilaenv(func() *int { y := 11; return &y }(), []byte("DSTEMR"), []byte("VA"), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }()) == 1 {
				//           Call DSTEMR, do test 27 (relative eigenvalue accuracy)
				//
				//              If S is positive definite and diagonally dominant,
				//              ask for all eigenvalues with high relative accuracy.
				vl = zero
				vu = zero
				il = 0
				iu = 0
				if jtype == 21 && srel {
					ntest = 27
					abstol = unfl + unfl
					golapack.Dstemr('V', 'A', &n, sd, se, &vl, &vu, &il, &iu, &m, wr, z, ldu, &n, iwork, &tryrac, work, lwork, toSlice(iwork, 2*n), toPtr((*lwork)-2*n), &iinfo)
					if iinfo != 0 {
						fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(V,A,rel)", iinfo, n, jtype, ioldsd)
						(*info) = abs(iinfo)
						if iinfo < 0 {
							return
						} else {
							result.Set(26, ulpinv)
							goto label270
						}
					}

					//              Do test 27
					temp2 = two * (two*float64(n) - one) * ulp * (one + eight*math.Pow(half, 2)) / math.Pow(one-half, 4)

					temp1 = zero
					for j = 1; j <= n; j++ {
						temp1 = math.Max(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j))/(abstol+math.Abs(d4.Get(j-1))))
					}

					result.Set(26, temp1/temp2)

					il = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}

					if srange {
						ntest = 28
						abstol = unfl + unfl
						golapack.Dstemr('V', 'I', &n, sd, se, &vl, &vu, &il, &iu, &m, wr, z, ldu, &n, iwork, &tryrac, work, lwork, toSlice(iwork, 2*n), toPtr((*lwork)-2*n), &iinfo)

						if iinfo != 0 {
							fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(V,I,rel)", iinfo, n, jtype, ioldsd)
							(*info) = abs(iinfo)
							if iinfo < 0 {
								return
							} else {
								result.Set(27, ulpinv)
								goto label270
							}
						}

						//                 Do test 28
						temp2 = two * (two*float64(n) - one) * ulp * (one + eight*math.Pow(half, 2)) / math.Pow(one-half, 4)

						temp1 = zero
						for j = il; j <= iu; j++ {
							temp1 = math.Max(temp1, math.Abs(wr.Get(j-il)-d4.Get(n-j))/(abstol+math.Abs(wr.Get(j-il))))
						}

						result.Set(27, temp1/temp2)
					} else {
						result.Set(27, zero)
					}
				} else {
					result.Set(26, zero)
					result.Set(27, zero)
				}

				//           Call DSTEMR(V,I) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}
				golapack.Dlaset('F', &n, &n, &zero, &one, z, ldu)

				if srange {
					ntest = 29
					il = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}
					golapack.Dstemr('V', 'I', &n, d5, work, &vl, &vu, &il, &iu, &m, d1, z, ldu, &n, iwork, &tryrac, work.Off(n), toPtr((*lwork)-n), toSlice(iwork, 2*n), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(V,I)", iinfo, n, jtype, ioldsd)
						(*info) = abs(iinfo)
						if iinfo < 0 {
							return
						} else {
							result.Set(28, ulpinv)
							goto label280
						}
					}

					//           Do Tests 29 and 30
					Dstt22(&n, &m, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work.Matrix(m, opts), &m, result.Off(28))

					//           Call DSTEMR to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
					}

					ntest = 31
					golapack.Dstemr('N', 'I', &n, d5, work, &vl, &vu, &il, &iu, &m, d2, z, ldu, &n, iwork, &tryrac, work.Off(n), toPtr((*lwork)-n), toSlice(iwork, 2*n), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(N,I)", iinfo, n, jtype, ioldsd)
						(*info) = abs(iinfo)
						if iinfo < 0 {
							return
						} else {
							result.Set(30, ulpinv)
							goto label280
						}
					}

					//           Do Test 31
					temp1 = zero
					temp2 = zero

					for j = 1; j <= iu-il+1; j++ {
						temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
						temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(30, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

					//           Call DSTEMR(V,V) to compute D1 and Z, do tests.
					//
					//           Compute D1 and Z
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
					}
					golapack.Dlaset('F', &n, &n, &zero, &one, z, ldu)

					ntest = 32

					if n > 0 {
						if il != 1 {
							vl = d2.Get(il-1) - math.Max(half*(d2.Get(il-1)-d2.Get(il-1-1)), math.Max(ulp*anorm, two*rtunfl))
						} else {
							vl = d2.Get(0) - math.Max(half*(d2.Get(n-1)-d2.Get(0)), math.Max(ulp*anorm, two*rtunfl))
						}
						if iu != n {
							vu = d2.Get(iu-1) + math.Max(half*(d2.Get(iu)-d2.Get(iu-1)), math.Max(ulp*anorm, two*rtunfl))
						} else {
							vu = d2.Get(n-1) + math.Max(half*(d2.Get(n-1)-d2.Get(0)), math.Max(ulp*anorm, two*rtunfl))
						}
					} else {
						vl = zero
						vu = one
					}

					golapack.Dstemr('V', 'V', &n, d5, work, &vl, &vu, &il, &iu, &m, d1, z, ldu, &n, iwork, &tryrac, work.Off(n), toPtr((*lwork)-n), toSlice(iwork, 2*n), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(V,V)", iinfo, n, jtype, ioldsd)
						(*info) = abs(iinfo)
						if iinfo < 0 {
							return
						} else {
							result.Set(31, ulpinv)
							goto label280
						}
					}

					//           Do Tests 32 and 33
					Dstt22(&n, &m, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work.Matrix(m, opts), &m, result.Off(31))

					//           Call DSTEMR to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
					}

					ntest = 34
					golapack.Dstemr('N', 'V', &n, d5, work, &vl, &vu, &il, &iu, &m, d2, z, ldu, &n, iwork, &tryrac, work.Off(n), toPtr((*lwork)-n), toSlice(iwork, 2*n), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(N,V)", iinfo, n, jtype, ioldsd)
						(*info) = abs(iinfo)
						if iinfo < 0 {
							return
						} else {
							result.Set(33, ulpinv)
							goto label280
						}
					}

					//           Do Test 34
					temp1 = zero
					temp2 = zero

					for j = 1; j <= iu-il+1; j++ {
						temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
						temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(33, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
				} else {
					result.Set(28, zero)
					result.Set(29, zero)
					result.Set(30, zero)
					result.Set(31, zero)
					result.Set(32, zero)
					result.Set(33, zero)
				}

				//           Call DSTEMR(V,A) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				ntest = 35

				golapack.Dstemr('V', 'A', &n, d5, work, &vl, &vu, &il, &iu, &m, d1, z, ldu, &n, iwork, &tryrac, work.Off(n), toPtr((*lwork)-n), toSlice(iwork, 2*n), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(V,A)", iinfo, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(34, ulpinv)
						goto label280
					}
				}

				//           Do Tests 35 and 36
				Dstt22(&n, &m, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work.Matrix(m, opts), &m, result.Off(34))

				//           Call DSTEMR to compute D2, do tests.
				//
				//           Compute D2
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				ntest = 37
				golapack.Dstemr('N', 'A', &n, d5, work, &vl, &vu, &il, &iu, &m, d2, z, ldu, &n, iwork, &tryrac, work.Off(n), toPtr((*lwork)-n), toSlice(iwork, 2*n), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEMR(N,A)", iinfo, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(36, ulpinv)
						goto label280
					}
				}

				//           Do Test 34
				temp1 = zero
				temp2 = zero

				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
				}

				result.Set(36, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			}
		label270:
			;
		label280:
			;
			ntestt = ntestt + ntest

			//           End of Loop -- Check for RESULT(j) > THRESH
			//
			//
			//           Print out tests which fail.
			for jr = 1; jr <= ntest; jr++ {
				if result.Get(jr-1) >= (*thresh) {
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Real Symmetric eigenvalue problem\n", "DST")
						fmt.Printf(" Matrix types (see DCHKST2STG for details): \n")
						fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
						fmt.Printf(" Dense %s Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Symmetric")
						fmt.Printf(" 16=Positive definite, evenly spaced eigenvalues\n 17=Positive definite, geometrically spaced eigenvlaues\n 18=Positive definite, clustered eigenvalues\n 19=Positive definite, small evenly spaced eigenvalues\n 20=Positive definite, large evenly spaced eigenvalues\n 21=Diagonally dominant tridiagonal, geometrically spaced eigenvalues\n")

						//                    Tests performed
						fmt.Printf("\nTest performed:  see DCHKST2STG for details.\n\n")
					}
					nerrs = nerrs + 1
					fmt.Printf(" N=%5d, seed=%4d, type %2d, test(%2d)=%10.3f\n", n, ioldsd, jtype, jr, result.Get(jr-1))
				}
			}
		label300:
		}
	}

	//     Summary
	Dlasum([]byte("DST"), &nerrs, &ntestt)
}
