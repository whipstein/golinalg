package eig

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkst2stg checks the Hermitian eigenvalue problem routines
// using the 2-stage reduction techniques. Since the generation
// of Q or the vectors is not available in this release, we only
// compare the eigenvalue resulting when using the 2-stage to the
// one considered as reference using the standard 1-stage reduction
// ZHETRD. For that, we call the standard ZHETRD and compute D1 using
// DSTEQR, then we call the 2-stage ZHETRD_2STAGE with Upper and Lower
// and we compute D2 and D3 using DSTEQR and then we replaced tests
// 3 and 4 by tests 11 and 12. test 1 and 2 remain to verify that
// the 1-stage results are OK and can be trusted.
// This testing routine will converge to the ZCHKST in the next
// release when vectors and generation of Q will be implemented.
//
//    ZHETRD factors A as  U S U* , where * means conjugate transpose,
//    S is real symmetric tridiagonal, and U is unitary.
//    ZHETRD can use either just the lower or just the upper triangle
//    of A; ZCHKST2STG checks both cases.
//    U is represented as a product of Householder
//    transformations, whose vectors are stored in the first
//    n-1 columns of V, and whose scale factors are in TAU.
//
//    ZHPTRD does the same as ZHETRD, except that A and V are stored
//    in "packed" format.
//
//    ZUNGTR constructs the matrix U from the contents of V and TAU.
//
//    ZUPGTR constructs the matrix U from the contents of VP and TAU.
//
//    ZSTEQR factors S as  Z D1 Z* , where Z is the unitary
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal.  D2 is the matrix of
//    eigenvalues computed when Z is not computed.
//
//    DSTERF computes D3, the matrix of eigenvalues, by the
//    PWK method, which does not yield eigenvectors.
//
//    ZPTEQR factors S as  Z4 D4 Z4* , for a
//    Hermitian positive definite tridiagonal matrix.
//    D5 is the matrix of eigenvalues computed when Z is not
//    computed.
//
//    DSTEBZ computes selected eigenvalues.  WA1, WA2, and
//    WA3 will denote eigenvalues computed to high
//    absolute accuracy, with different range options.
//    WR will denote eigenvalues computed to high relative
//    accuracy.
//
//    ZSTEIN computes Y, the eigenvectors of S, given the
//    eigenvalues.
//
//    ZSTEDC factors S as Z D1 Z* , where Z is the unitary
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option). It may also
//    update an input unitary matrix, usually the output
//    from ZHETRD/ZUNGTR or ZHPTRD/ZUPGTR ('V' option). It may
//    also just compute eigenvalues ('N' option).
//
//    ZSTEMR factors S as Z D1 Z* , where Z is the unitary
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option).  ZSTEMR
//    uses the Relatively Robust Representation whenever possible.
//
// When ZCHKST2STG is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each _type of matrix, one matrix will be generated and used
// to test the Hermitian eigenroutines.  For each matrix, a number
// of tests will be performed:
//
// (1)     | A - V S V* | / ( |A| n ulp ) ZHETRD( UPLO='U', ... )
//
// (2)     | I - UV* | / ( n ulp )        ZUNGTR( UPLO='U', ... )
//
// (3)     | A - V S V* | / ( |A| n ulp ) ZHETRD( UPLO='L', ... )
//         replaced by | D1 - D2 | / ( |D1| ulp ) where D1 is the
//         eigenvalue matrix computed using S and D2 is the
//         eigenvalue matrix computed using S_2stage the output of
//         ZHETRD_2STAGE("N", "U",....). D1 and D2 are computed
//         via DSTEQR('N',...)
//
// (4)     | I - UV* | / ( n ulp )        ZUNGTR( UPLO='L', ... )
//         replaced by | D1 - D3 | / ( |D1| ulp ) where D1 is the
//         eigenvalue matrix computed using S and D3 is the
//         eigenvalue matrix computed using S_2stage the output of
//         ZHETRD_2STAGE("N", "L",....). D1 and D3 are computed
//         via DSTEQR('N',...)
//
// (5-8)   Same as 1-4, but for ZHPTRD and ZUPGTR.
//
// (9)     | S - Z D Z* | / ( |S| n ulp ) ZSTEQR('V',...)
//
// (10)    | I - ZZ* | / ( n ulp )        ZSTEQR('V',...)
//
// (11)    | D1 - D2 | / ( |D1| ulp )        ZSTEQR('N',...)
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
// (14)    | S - Z4 D4 Z4* | / ( |S| n ulp ) ZPTEQR('V',...)
//
// (15)    | I - Z4 Z4* | / ( n ulp )        ZPTEQR('V',...)
//
// (16)    | D4 - D5 | / ( 100 |D4| ulp )       ZPTEQR('N',...)
//
// When S is also diagonally dominant by the factor gamma < 1,
//
// (17)    max | D4(i) - WR(i) | / ( |D4(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              DSTEBZ( 'A', 'E', ...)
//
// (18)    | WA1 - D3 | / ( |D3| ulp )          DSTEBZ( 'A', 'E', ...)
//
// (19)    ( max { minint | WA2(i)-WA3(j) | } +
//            i     j
//           max { minint | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//                                              DSTEBZ( 'I', 'E', ...)
//
// (20)    | S - Y WA1 Y* | / ( |S| n ulp )  DSTEBZ, ZSTEIN
//
// (21)    | I - Y Y* | / ( n ulp )          DSTEBZ, ZSTEIN
//
// (22)    | S - Z D Z* | / ( |S| n ulp )    ZSTEDC('I')
//
// (23)    | I - ZZ* | / ( n ulp )           ZSTEDC('I')
//
// (24)    | S - Z D Z* | / ( |S| n ulp )    ZSTEDC('V')
//
// (25)    | I - ZZ* | / ( n ulp )           ZSTEDC('V')
//
// (26)    | D1 - D2 | / ( |D1| ulp )           ZSTEDC('V') and
//                                              ZSTEDC('N')
//
// Test 27 is disabled at the moment because ZSTEMR does not
// guarantee high relatvie accuracy.
//
// (27)    max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              ZSTEMR('V', 'A')
//
// (28)    max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              ZSTEMR('V', 'I')
//
// Tests 29 through 34 are disable at present because ZSTEMR
// does not handle partial spectrum requests.
//
// (29)    | S - Z D Z* | / ( |S| n ulp )    ZSTEMR('V', 'I')
//
// (30)    | I - ZZ* | / ( n ulp )           ZSTEMR('V', 'I')
//
// (31)    ( max { minint | WA2(i)-WA3(j) | } +
//            i     j
//           max { minint | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         ZSTEMR('N', 'I') vs. CSTEMR('V', 'I')
//
// (32)    | S - Z D Z* | / ( |S| n ulp )    ZSTEMR('V', 'V')
//
// (33)    | I - ZZ* | / ( n ulp )           ZSTEMR('V', 'V')
//
// (34)    ( max { minint | WA2(i)-WA3(j) | } +
//            i     j
//           max { minint | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         ZSTEMR('N', 'V') vs. CSTEMR('V', 'V')
//
// (35)    | S - Z D Z* | / ( |S| n ulp )    ZSTEMR('V', 'A')
//
// (36)    | I - ZZ* | / ( n ulp )           ZSTEMR('V', 'A')
//
// (37)    ( max { minint | WA2(i)-WA3(j) | } +
//            i     j
//           max { minint | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         ZSTEMR('N', 'A') vs. CSTEMR('V', 'A')
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
// (16) Same as (8), but diagonal elements are all positive.
// (17) Same as (9), but diagonal elements are all positive.
// (18) Same as (10), but diagonal elements are all positive.
// (19) Same as (16), but multiplied by SQRT( overflow threshold )
// (20) Same as (16), but multiplied by SQRT( underflow threshold )
// (21) A diagonally dominant tridiagonal matrix with geometrically
//      spaced diagonal entries 1, ..., ULP.
func Zchkst2stg(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.CMatrix, lda *int, ap *mat.CVector, sd, se, d1, d2, d3, d4, d5, wa1, wa2, wa3, wr *mat.Vector, u *mat.CMatrix, ldu *int, v *mat.CMatrix, vp, tau *mat.CVector, z *mat.CMatrix, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork *int, result *mat.Vector, info *int, t *testing.T) {
	var badnn, crange, crel, tryrac bool
	var cone, czero complex128
	var abstol, aninv, anorm, cond, eight, half, hun, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, temp4, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, il, imode, inde, indrwk, itemp, itype, iu, j, jc, jr, jsize, jtype, lgn, lh, liwedc, log2ui, lrwedc, lw, lwedc, m, m2, m3, maxtyp, mtypes, n, nap, nblock, nerrs, nmats, nmax, nsplit, ntest, ntestt int
	dumma := vf(1)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0
	ten = 10.0
	hun = 100.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	half = one / two
	maxtyp = 21
	crange = false
	crel = false

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
		nmax = maxint(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
	}

	nblock = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZHETRD"), []byte{'L'}, &nmax, toPtr(-1), toPtr(-1), toPtr(-1))
	nblock = minint(nmax, maxint(1, nblock))

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
	} else if 2*powint(maxint(2, nmax), 2) > (*lwork) {
		(*info) = -29
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZCHKST2STG"), -(*info))
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
			if powint(2, lgn) < n {
				lgn = lgn + 1
			}
			if powint(2, lgn) < n {
				lgn = lgn + 1
			}
			lwedc = 1 + 4*n + 2*n*lgn + 4*powint(n, 2)
			lrwedc = 1 + 3*n + 2*n*lgn + 4*powint(n, 2)
			liwedc = 6 + 6*n + 5*n*lgn
		} else {
			lwedc = 8
			lrwedc = 7
			liwedc = 12
		}
		nap = (n * (n + 1)) / 2
		aninv = one / float64(maxint(1, n))
		//
		if (*nsizes) != 1 {
			mtypes = minint(maxtyp, *ntypes)
		} else {
			mtypes = minint(maxtyp+1, *ntypes)
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
			//           =5         random log   Hermitian, w/ eigenvalues
			//           =6         random       (none)
			//           =7                      random diagonal
			//           =8                      random Hermitian
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

			golapack.Zlaset('F', lda, &n, &czero, &czero, a, lda)
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
					a.SetRe(jc-1, jc-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work, &iinfo)

			} else if itype == 5 {
				//              Hermitian, eigenvalues specified
				matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work, &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Zlatmr(&n, &n, 'S', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Hermitian, random eigenvalues
				matgen.Zlatmr(&n, &n, 'S', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              Positive definite, eigenvalues specified.
				matgen.Zlatms(&n, &n, 'S', iseed, 'P', rwork, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work, &iinfo)

			} else if itype == 10 {
				//              Positive definite tridiagonal, eigenvalues specified.
				matgen.Zlatms(&n, &n, 'S', iseed, 'P', rwork, &imode, &cond, &anorm, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), 'N', a, lda, work, &iinfo)
				for i = 2; i <= n; i++ {
					temp1 = a.GetMag(i-1-1, i-1)
					temp2 = math.Sqrt(cmplx.Abs(a.Get(i-1-1, i-1-1) * a.Get(i-1, i-1)))
					if temp1 > half*temp2 {
						a.Set(i-1-1, i-1, a.Get(i-1-1, i-1)*complex((half*temp2/(unfl+temp1)), 0))
						a.Set(i-1, i-1-1, a.GetConj(i-1-1, i-1))
					}
				}

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				return
			}

		label100:
			;

			//           Call ZHETRD and ZUNGTR to compute S and U from
			//           upper triangle.
			golapack.Zlacpy('U', &n, &n, a, lda, v, ldu)

			ntest = 1
			golapack.Zhetrd('U', &n, v, ldu, sd, se, tau, work, lwork, &iinfo)

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHETRD(U)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(0, ulpinv)
					goto label280
				}
			}

			golapack.Zlacpy('U', &n, &n, v, ldu, u, ldu)

			ntest = 2
			golapack.Zungtr('U', &n, u, ldu, tau, work, lwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUNGTR(U)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(1, ulpinv)
					goto label280
				}
			}

			//           Do tests 1 and 2
			Zhet21(func() *int { y := 2; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), a, lda, sd, se, u, ldu, v, ldu, tau, work, rwork, result.Off(0))
			Zhet21(func() *int { y := 3; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), a, lda, sd, se, u, ldu, v, ldu, tau, work, rwork, result.Off(1))

			//           Compute D1 the eigenvalues resulting from the tridiagonal
			//           form using the standard 1-stage algorithm and use it as a
			//           reference to compare with the 2-stage technique
			//
			//           Compute D1 from the 1-stage and used as reference for the
			//           2-stage
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d1, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
			}

			golapack.Zsteqr('N', &n, d1, rwork, work.CMatrix(*ldu, opts), ldu, rwork.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
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
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, sd.Matrix(1, opts), func() *int { y := 1; return &y }())
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, se.Matrix(1, opts), func() *int { y := 1; return &y }())
			golapack.Zlacpy('U', &n, &n, a, lda, v, ldu)
			lh = maxint(1, 4*n)
			lw = (*lwork) - lh
			golapack.Zhetrd2stage('N', 'U', &n, v, ldu, sd, se, tau, work, &lh, work.Off(lh+1-1), &lw, &iinfo)

			//           Compute D2 from the 2-stage Upper case
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d2, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
			}

			ntest = 3
			golapack.Zsteqr('N', &n, d2, rwork, work.CMatrix(*ldu, opts), ldu, rwork.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
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
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, sd.Matrix(1, opts), func() *int { y := 1; return &y }())
			golapack.Dlaset('F', &n, func() *int { y := 1; return &y }(), &zero, &zero, se.Matrix(1, opts), func() *int { y := 1; return &y }())
			golapack.Zlacpy('L', &n, &n, a, lda, v, ldu)
			golapack.Zhetrd2stage('N', 'L', &n, v, ldu, sd, se, tau, work, &lh, work.Off(lh+1-1), &lw, &iinfo)

			//           Compute D3 from the 2-stage Upper case
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d3, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
			}

			ntest = 4
			golapack.Zsteqr('N', &n, d3, rwork, work.CMatrix(*ldu, opts), ldu, rwork.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
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
				temp1 = maxf64(temp1, d1.GetMag(j-1), d2.GetMag(j-1))
				temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
				temp3 = maxf64(temp3, d1.GetMag(j-1), d3.GetMag(j-1))
				temp4 = maxf64(temp4, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
			}

			result.Set(2, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))
			result.Set(3, temp4/maxf64(unfl, ulp*maxf64(temp3, temp4)))

			//           Store the upper triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = 1; jr <= jc; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call ZHPTRD and ZUPGTR to compute S and U from AP
			goblas.Zcopy(&nap, ap, func() *int { y := 1; return &y }(), vp, func() *int { y := 1; return &y }())

			ntest = 5
			golapack.Zhptrd('U', &n, vp, sd, se, tau, &iinfo)

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHPTRD(U)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(4, ulpinv)
					goto label280
				}
			}

			ntest = 6
			golapack.Zupgtr('U', &n, vp, tau, u, ldu, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUPGTR(U)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(5, ulpinv)
					goto label280
				}
			}

			//           Do tests 5 and 6
			Zhpt21(func() *int { y := 2; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, rwork, result.Off(4))
			Zhpt21(func() *int { y := 3; return &y }(), 'U', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, rwork, result.Off(5))

			//           Store the lower triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = jc; jr <= n; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call ZHPTRD and ZUPGTR to compute S and U from AP
			goblas.Zcopy(&nap, ap, func() *int { y := 1; return &y }(), vp, func() *int { y := 1; return &y }())

			ntest = 7
			golapack.Zhptrd('L', &n, vp, sd, se, tau, &iinfo)

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHPTRD(L)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(6, ulpinv)
					goto label280
				}
			}

			ntest = 8
			golapack.Zupgtr('L', &n, vp, tau, u, ldu, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUPGTR(L)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(7, ulpinv)
					goto label280
				}
			}

			Zhpt21(func() *int { y := 2; return &y }(), 'L', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, rwork, result.Off(6))
			Zhpt21(func() *int { y := 3; return &y }(), 'L', &n, func() *int { y := 1; return &y }(), ap, sd, se, u, ldu, vp, tau, work, rwork, result.Off(7))

			//           Call ZSTEQR to compute D1, D2, and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d1, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
			}
			golapack.Zlaset('F', &n, &n, &czero, &cone, z, ldu)

			ntest = 9
			golapack.Zsteqr('V', &n, d1, rwork, z, ldu, rwork.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(V)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(8, ulpinv)
					goto label280
				}
			}

			//           Compute D2
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d2, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
			}

			ntest = 11
			golapack.Zsteqr('N', &n, d2, rwork, work.CMatrix(*ldu, opts), ldu, rwork.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEQR(N)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(10, ulpinv)
					goto label280
				}
			}

			//           Compute D3 (using PWK method)
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d3, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
			}

			ntest = 12
			golapack.Dsterf(&n, d3, rwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTERF", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(11, ulpinv)
					goto label280
				}
			}

			//           Do Tests 9 and 10
			Zstt21(&n, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work, rwork, result.Off(8))

			//           Do Tests 11 and 12
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

			result.Set(10, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))
			result.Set(11, temp4/maxf64(unfl, ulp*maxf64(temp3, temp4)))

			//           Do Test 13 -- Sturm Sequence Test of Eigenvalues
			//                         Go up by factors of two until it succeeds
			ntest = 13
			temp1 = (*thresh) * (half - ulp)

			for j = 0; j <= log2ui; j++ {
				Dstech(&n, sd, se, d1, &temp1, rwork, &iinfo)
				if iinfo == 0 {
					goto label170
				}
				temp1 = temp1 * two
			}

		label170:
			;
			result.Set(12, temp1)

			//           For positive definite matrices ( JTYPE.GT.15 ) call ZPTEQR
			//           and do tests 14, 15, and 16 .
			if jtype > 15 {
				//              Compute D4 and Z4
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d4, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}
				golapack.Zlaset('F', &n, &n, &czero, &cone, z, ldu)

				ntest = 14
				golapack.Zpteqr('V', &n, d4, rwork, z, ldu, rwork.Off(n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZPTEQR(V)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(13, ulpinv)
						goto label280
					}
				}

				//              Do Tests 14 and 15
				Zstt21(&n, func() *int { y := 0; return &y }(), sd, se, d4, dumma, z, ldu, work, rwork, result.Off(13))

				//              Compute D5
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d5, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}

				ntest = 16
				golapack.Zpteqr('N', &n, d5, rwork, z, ldu, rwork.Off(n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZPTEQR(N)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
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
					temp1 = maxf64(temp1, d4.GetMag(j-1), d5.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d4.Get(j-1)-d5.Get(j-1)))
				}

				result.Set(15, temp2/maxf64(unfl, hun*ulp*maxf64(temp1, temp2)))
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
				golapack.Dstebz('A', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m, &nsplit, wr, iwork, toSlice(iwork, n+1-1), rwork, toSlice(iwork, 2*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(A,rel)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
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
					temp1 = maxf64(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j+1-1))/(abstol+math.Abs(d4.Get(j-1))))
				}

				result.Set(16, temp1/temp2)
			} else {
				result.Set(16, zero)
			}

			//           Now ask for all eigenvalues with high absolute accuracy.
			ntest = 18
			abstol = unfl + unfl
			golapack.Dstebz('A', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m, &nsplit, wa1, iwork, toSlice(iwork, n+1-1), rwork, toSlice(iwork, 2*n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(A)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
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
				temp1 = maxf64(temp1, d3.GetMag(j-1), wa1.GetMag(j-1))
				temp2 = maxf64(temp2, math.Abs(d3.Get(j-1)-wa1.Get(j-1)))
			}

			result.Set(17, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

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
			//
			golapack.Dstebz('I', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m2, &nsplit, wa2, iwork, toSlice(iwork, n+1-1), rwork, toSlice(iwork, 2*n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(I)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
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
					vl = wa1.Get(il-1) - maxf64(half*(wa1.Get(il-1)-wa1.Get(il-1-1)), ulp*anorm, two*rtunfl)
				} else {
					vl = wa1.Get(0) - maxf64(half*(wa1.Get(n-1)-wa1.Get(0)), ulp*anorm, two*rtunfl)
				}
				if iu != n {
					vu = wa1.Get(iu-1) + maxf64(half*(wa1.Get(iu+1-1)-wa1.Get(iu-1)), ulp*anorm, two*rtunfl)
				} else {
					vu = wa1.Get(n-1) + maxf64(half*(wa1.Get(n-1)-wa1.Get(0)), ulp*anorm, two*rtunfl)
				}
			} else {
				vl = zero
				vu = one
			}

			golapack.Dstebz('V', 'E', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m3, &nsplit, wa3, iwork, toSlice(iwork, n+1-1), rwork, toSlice(iwork, 2*n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(V)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
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
				temp3 = maxf64(wa1.GetMag(n-1), wa1.GetMag(0))
			} else {
				temp3 = zero
			}

			result.Set(18, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			//           Call ZSTEIN to compute eigenvectors corresponding to
			//           eigenvalues in WA1.  (First call DSTEBZ again, to make sure
			//           it returns these eigenvalues in the correct order.)
			ntest = 21
			golapack.Dstebz('A', 'B', &n, &vl, &vu, &il, &iu, &abstol, sd, se, &m, &nsplit, wa1, iwork, toSlice(iwork, n+1-1), rwork, toSlice(iwork, 2*n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEBZ(A,B)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			golapack.Zstein(&n, sd, se, &m, wa1, iwork, toSlice(iwork, n+1-1), z, ldu, rwork, toSlice(iwork, 2*n+1-1), toSlice(iwork, 3*n+1-1), &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEIN", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			//           Do tests 20 and 21
			Zstt21(&n, func() *int { y := 0; return &y }(), sd, se, wa1, dumma, z, ldu, work, rwork, result.Off(19))

			//           Call ZSTEDC(I) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			inde = 1
			indrwk = inde + n
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d1, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork.Off(inde-1), func() *int { y := 1; return &y }())
			}
			golapack.Zlaset('F', &n, &n, &czero, &cone, z, ldu)

			ntest = 22
			golapack.Zstedc('I', &n, d1, rwork.Off(inde-1), z, ldu, work, &lwedc, rwork.Off(indrwk-1), &lrwedc, iwork, &liwedc, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEDC(I)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(21, ulpinv)
					goto label280
				}
			}

			//           Do Tests 22 and 23
			Zstt21(&n, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work, rwork, result.Off(21))

			//           Call ZSTEDC(V) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d1, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork.Off(inde-1), func() *int { y := 1; return &y }())
			}
			golapack.Zlaset('F', &n, &n, &czero, &cone, z, ldu)

			ntest = 24
			golapack.Zstedc('V', &n, d1, rwork.Off(inde-1), z, ldu, work, &lwedc, rwork.Off(indrwk-1), &lrwedc, iwork, &liwedc, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEDC(V)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(23, ulpinv)
					goto label280
				}
			}

			//           Do Tests 24 and 25
			Zstt21(&n, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work, rwork, result.Off(23))

			//           Call ZSTEDC(N) to compute D2, do tests.
			//
			//           Compute D2
			goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d2, func() *int { y := 1; return &y }())
			if n > 0 {
				goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork.Off(inde-1), func() *int { y := 1; return &y }())
			}
			golapack.Zlaset('F', &n, &n, &czero, &cone, z, ldu)
			//
			ntest = 26
			golapack.Zstedc('N', &n, d2, rwork.Off(inde-1), z, ldu, work, &lwedc, rwork.Off(indrwk-1), &lrwedc, iwork, &liwedc, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEDC(N)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
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
				temp1 = maxf64(temp1, d1.GetMag(j-1), d2.GetMag(j-1))
				temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
			}

			result.Set(25, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			//           Only test ZSTEMR if IEEE compliant
			if Ilaenv(func() *int { y := 10; return &y }(), []byte("ZSTEMR"), []byte("VA"), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }()) == 1 && Ilaenv(func() *int { y := 11; return &y }(), []byte("ZSTEMR"), []byte("VA"), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }()) == 1 {
				//           Call ZSTEMR, do test 27 (relative eigenvalue accuracy)
				//
				//              If S is positive definite and diagonally dominant,
				//              ask for all eigenvalues with high relative accuracy.
				vl = zero
				vu = zero
				il = 0
				iu = 0
				if jtype == 21 && crel {
					ntest = 27
					abstol = unfl + unfl
					golapack.Zstemr('V', 'A', &n, sd, se, &vl, &vu, &il, &iu, &m, wr, z, ldu, &n, iwork, &tryrac, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*lwork)-2*n), &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(V,A,rel)", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
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
						temp1 = maxf64(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j+1-1))/(abstol+d4.GetMag(j-1)))
					}

					result.Set(26, temp1/temp2)

					il = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}

					if crange {
						ntest = 28
						abstol = unfl + unfl
						golapack.Zstemr('V', 'I', &n, sd, se, &vl, &vu, &il, &iu, &m, wr, z, ldu, &n, iwork, &tryrac, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*lwork)-2*n), &iinfo)

						if iinfo != 0 {
							t.Fail()
							fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(V,I,rel)", iinfo, n, jtype, ioldsd)
							(*info) = absint(iinfo)
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
							temp1 = maxf64(temp1, math.Abs(wr.Get(j-il+1-1)-d4.Get(n-j+1-1))/(abstol+math.Abs(wr.Get(j-il+1-1))))
						}

						result.Set(27, temp1/temp2)
					} else {
						result.Set(27, zero)
					}
				} else {
					result.Set(26, zero)
					result.Set(27, zero)
				}

				//           Call ZSTEMR(V,I) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d5, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}
				golapack.Zlaset('F', &n, &n, &czero, &cone, z, ldu)

				if crange {
					ntest = 29
					il = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}
					golapack.Zstemr('V', 'I', &n, d5, rwork, &vl, &vu, &il, &iu, &m, d1, z, ldu, &n, iwork, &tryrac, rwork.Off(n+1-1), toPtr((*lrwork)-n), toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(V,I)", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
						if iinfo < 0 {
							return
						} else {
							result.Set(28, ulpinv)
							goto label280
						}
					}

					//           Do Tests 29 and 30
					//
					//
					//           Call ZSTEMR to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d5, func() *int { y := 1; return &y }())
					if n > 0 {
						goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
					}

					ntest = 31
					golapack.Zstemr('N', 'I', &n, d5, rwork, &vl, &vu, &il, &iu, &m, d2, z, ldu, &n, iwork, &tryrac, rwork.Off(n+1-1), toPtr((*lrwork)-n), toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(N,I)", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
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
						temp1 = maxf64(temp1, d1.GetMag(j-1), d2.GetMag(j-1))
						temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(30, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

					//           Call ZSTEMR(V,V) to compute D1 and Z, do tests.
					//
					//           Compute D1 and Z
					goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d5, func() *int { y := 1; return &y }())
					if n > 0 {
						goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
					}
					golapack.Zlaset('F', &n, &n, &czero, &cone, z, ldu)

					ntest = 32

					if n > 0 {
						if il != 1 {
							vl = d2.Get(il-1) - maxf64(half*(d2.Get(il-1)-d2.Get(il-1-1)), ulp*anorm, two*rtunfl)
						} else {
							vl = d2.Get(0) - maxf64(half*(d2.Get(n-1)-d2.Get(0)), ulp*anorm, two*rtunfl)
						}
						if iu != n {
							vu = d2.Get(iu-1) + maxf64(half*(d2.Get(iu+1-1)-d2.Get(iu-1)), ulp*anorm, two*rtunfl)
						} else {
							vu = d2.Get(n-1) + maxf64(half*(d2.Get(n-1)-d2.Get(0)), ulp*anorm, two*rtunfl)
						}
					} else {
						vl = zero
						vu = one
					}

					golapack.Zstemr('V', 'V', &n, d5, rwork, &vl, &vu, &il, &iu, &m, d1, z, ldu, &m, iwork, &tryrac, rwork.Off(n+1-1), toPtr((*lrwork)-n), toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(V,V)", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
						if iinfo < 0 {
							return
						} else {
							result.Set(31, ulpinv)
							goto label280
						}
					}

					//           Do Tests 32 and 33
					Zstt22(&n, &m, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work.CMatrix(m, opts), &m, rwork, result.Off(31))

					//           Call ZSTEMR to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d5, func() *int { y := 1; return &y }())
					if n > 0 {
						goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
					}

					ntest = 34
					golapack.Zstemr('N', 'V', &n, d5, rwork, &vl, &vu, &il, &iu, &m, d2, z, ldu, &n, iwork, &tryrac, rwork.Off(n+1-1), toPtr((*lrwork)-n), toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(N,V)", iinfo, n, jtype, ioldsd)
						(*info) = absint(iinfo)
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
						temp1 = maxf64(temp1, d1.GetMag(j-1), d2.GetMag(j-1))
						temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(33, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))
				} else {
					result.Set(28, zero)
					result.Set(29, zero)
					result.Set(30, zero)
					result.Set(31, zero)
					result.Set(32, zero)
					result.Set(33, zero)
				}

				//           Call ZSTEMR(V,A) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d5, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}

				ntest = 35

				golapack.Zstemr('V', 'A', &n, d5, rwork, &vl, &vu, &il, &iu, &m, d1, z, ldu, &n, iwork, &tryrac, rwork.Off(n+1-1), toPtr((*lrwork)-n), toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(V,A)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(34, ulpinv)
						goto label280
					}
				}

				//           Do Tests 35 and 36
				Zstt22(&n, &m, func() *int { y := 0; return &y }(), sd, se, d1, dumma, z, ldu, work.CMatrix(m, opts), &m, rwork, result.Off(34))

				//           Call ZSTEMR to compute D2, do tests.
				//
				//           Compute D2
				goblas.Dcopy(&n, sd, func() *int { y := 1; return &y }(), d5, func() *int { y := 1; return &y }())
				if n > 0 {
					goblas.Dcopy(toPtr(n-1), se, func() *int { y := 1; return &y }(), rwork, func() *int { y := 1; return &y }())
				}

				ntest = 37
				golapack.Zstemr('N', 'A', &n, d5, rwork, &vl, &vu, &il, &iu, &m, d2, z, ldu, &n, iwork, &tryrac, rwork.Off(n+1-1), toPtr((*lrwork)-n), toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZSTEMR(N,A)", iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
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
					temp1 = maxf64(temp1, d1.GetMag(j-1), d2.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
				}

				result.Set(36, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))
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
					t.Fail()
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Complex Hermitian eigenvalue problem\n", "ZST")
						fmt.Printf(" Matrix types (see ZCHKST2STG for details): \n")
						fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
						fmt.Printf(" Dense %s Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Hermitian")
						fmt.Printf(" 16=Positive definite, evenly spaced eigenvalues\n 17=Positive definite, geometrically spaced eigenvlaues\n 18=Positive definite, clustered eigenvalues\n 19=Positive definite, small evenly spaced eigenvalues\n 20=Positive definite, large evenly spaced eigenvalues\n 21=Diagonally dominant tridiagonal, geometrically spaced eigenvalues\n")

						//                    Tests performed
						fmt.Printf("\nTest performed:  see ZCHKST2STG for details.\n\n")
					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}
		label300:
		}
	}

	//     Summary
	Dlasum([]byte("ZST"), &nerrs, &ntestt)
}
