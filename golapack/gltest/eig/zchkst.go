package eig

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchkst checks the Hermitian eigenvalue problem routines.
//
//    Zhetrd factors A as  U S U* , where * means conjugate transpose,
//    S is real symmetric tridiagonal, and U is unitary.
//    Zhetrd can use either just the lower or just the upper triangle
//    of A; zchkst checks both cases.
//    U is represented as a product of Householder
//    transformations, whose vectors are stored in the first
//    n-1 columns of V, and whose scale factors are in TAU.
//
//    Zhptrd does the same as Zhetrd, except that A and V are stored
//    in "packed" format.
//
//    Zungtr constructs the matrix U from the contents of V and TAU.
//
//    Zupgtr constructs the matrix U from the contents of VP and TAU.
//
//    Zsteqr factors S as  Z D1 Z* , where Z is the unitary
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal.  D2 is the matrix of
//    eigenvalues computed when Z is not computed.
//
//    Dsterf computes D3, the matrix of eigenvalues, by the
//    PWK method, which does not yield eigenvectors.
//
//    Zpteqr factors S as  Z4 D4 Z4* , for a
//    Hermitian positive definite tridiagonal matrix.
//    D5 is the matrix of eigenvalues computed when Z is not
//    computed.
//
//    Dstebz computes selected eigenvalues.  WA1, WA2, and
//    WA3 will denote eigenvalues computed to high
//    absolute accuracy, with different range options.
//    WR will denote eigenvalues computed to high relative
//    accuracy.
//
//    ZSTEIN computes Y, the eigenvectors of S, given the
//    eigenvalues.
//
//    Zstedc factors S as Z D1 Z* , where Z is the unitary
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option). It may also
//    update an input unitary matrix, usually the output
//    from Zhetrd/Zungtr or Zhptrd/Zupgtr ('V' option). It may
//    also just compute eigenvalues ('N' option).
//
//    Zstemr factors S as Z D1 Z* , where Z is the unitary
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option).  Zstemr
//    uses the Relatively Robust Representation whenever possible.
//
// When zchkst is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each _type of matrix, one matrix will be generated and used
// to test the Hermitian eigenroutines.  For each matrix, a number
// of tests will be performed:
//
// (1)     | A - V S V* | / ( |A| n ulp ) Zhetrd( UPLO='U', ... )
//
// (2)     | I - UV* | / ( n ulp )        Zungtr( UPLO='U', ... )
//
// (3)     | A - V S V* | / ( |A| n ulp ) Zhetrd( UPLO='L', ... )
//
// (4)     | I - UV* | / ( n ulp )        Zungtr( UPLO='L', ... )
//
// (5-8)   Same as 1-4, but for Zhptrd and Zupgtr.
//
// (9)     | S - Z D Z* | / ( |S| n ulp ) Zsteqr('V',...)
//
// (10)    | I - ZZ* | / ( n ulp )        Zsteqr('V',...)
//
// (11)    | D1 - D2 | / ( |D1| ulp )        Zsteqr('N',...)
//
// (12)    | D1 - D3 | / ( |D1| ulp )        Dsterf
//
// (13)    0 if the true eigenvalues (computed by sturm count)
//         of S are within THRESH of
//         those in D1.  2*THRESH if they are not.  (Tested using
//         DSTECH)
//
// For S positive definite,
//
// (14)    | S - Z4 D4 Z4* | / ( |S| n ulp ) Zpteqr('V',...)
//
// (15)    | I - Z4 Z4* | / ( n ulp )        Zpteqr('V',...)
//
// (16)    | D4 - D5 | / ( 100 |D4| ulp )       Zpteqr('N',...)
//
// When S is also diagonally dominant by the factor gamma < 1,
//
// (17)    max | D4(i) - WR(i) | / ( |D4(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              Dstebz( 'A', 'E', ...)
//
// (18)    | WA1 - D3 | / ( |D3| ulp )          Dstebz( 'A', 'E', ...)
//
// (19)    ( max { min | WA2(i)-WA3(j) | } +
//            i     j
//           max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//                                              Dstebz( 'I', 'E', ...)
//
// (20)    | S - Y WA1 Y* | / ( |S| n ulp )  Dstebz, ZSTEIN
//
// (21)    | I - Y Y* | / ( n ulp )          Dstebz, ZSTEIN
//
// (22)    | S - Z D Z* | / ( |S| n ulp )    Zstedc('I')
//
// (23)    | I - ZZ* | / ( n ulp )           Zstedc('I')
//
// (24)    | S - Z D Z* | / ( |S| n ulp )    Zstedc('V')
//
// (25)    | I - ZZ* | / ( n ulp )           Zstedc('V')
//
// (26)    | D1 - D2 | / ( |D1| ulp )           Zstedc('V') and
//                                              Zstedc('N')
//
// Test 27 is disabled at the moment because Zstemr does not
// guarantee high relatvie accuracy.
//
// (27)    max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              Zstemr('V', 'A')
//
// (28)    max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              Zstemr('V', 'I')
//
// Tests 29 through 34 are disable at present because Zstemr
// does not handle partial spectrum requests.
//
// (29)    | S - Z D Z* | / ( |S| n ulp )    Zstemr('V', 'I')
//
// (30)    | I - ZZ* | / ( n ulp )           Zstemr('V', 'I')
//
// (31)    ( max { min | WA2(i)-WA3(j) | } +
//            i     j
//           max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         Zstemr('N', 'I') vs. CSTEMR('V', 'I')
//
// (32)    | S - Z D Z* | / ( |S| n ulp )    Zstemr('V', 'V')
//
// (33)    | I - ZZ* | / ( n ulp )           Zstemr('V', 'V')
//
// (34)    ( max { min | WA2(i)-WA3(j) | } +
//            i     j
//           max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         Zstemr('N', 'V') vs. CSTEMR('V', 'V')
//
// (35)    | S - Z D Z* | / ( |S| n ulp )    Zstemr('V', 'A')
//
// (36)    | I - ZZ* | / ( n ulp )           Zstemr('V', 'A')
//
// (37)    ( max { min | WA2(i)-WA3(j) | } +
//            i     j
//           max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         Zstemr('N', 'A') vs. CSTEMR('V', 'A')
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
func zchkst(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a *mat.CMatrix, ap *mat.CVector, sd, se, d1, d2, d3, d4, d5, wa1, wa2, wa3, wr *mat.Vector, u, v *mat.CMatrix, vp, tau *mat.CVector, z *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork []int, liwork int, result *mat.Vector) (err error) {
	var badnn, crange, crel, tryrac bool
	var cone, czero complex128
	var abstol, aninv, anorm, cond, eight, half, hun, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, temp4, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, il, imode, inde, indrwk, itemp, itype, iu, j, jc, jr, jsize, jtype, lgn, liwedc, log2ui, lrwedc, lwedc, m, m2, m3, maxtyp, mtypes, n, nap, nblock, nerrs, nmats, nmax, ntest, ntestt int
	dumma := vf(1)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	kmagn := []int{1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1}
	kmode := []int{0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 3, 1, 4, 4, 3}
	ktype := []int{1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9, 9, 9, 10}

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

	//     Keep ftnchek happy
	idumma[0] = 1

	//     Check for errors
	ntestt = 0

	//     Important constants
	badnn = false
	tryrac = true
	nmax = 1
	for j = 1; j <= nsizes; j++ {
		nmax = max(nmax, nn[j-1])
		if nn[j-1] < 0 {
			badnn = true
		}
	}

	nblock = ilaenv(1, "Zhetrd", []byte{'L'}, nmax, -1, -1, -1)
	nblock = min(nmax, max(1, nblock))

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsizes=%v", nsizes)
	} else if badnn {
		err = fmt.Errorf("badnn: nn=%v", nn)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if a.Rows < nmax {
		err = fmt.Errorf("a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if u.Rows < nmax {
		err = fmt.Errorf("u.Rows < nmax: u.Rows=%v, nmax=%v", u.Rows, nmax)
	} else if 2*pow(max(2, nmax), 2) > lwork {
		err = fmt.Errorf("2*pow(max(2, nmax), 2) > lwork: nmax=%v, lwork=%v", nmax, lwork)
	}

	if err != nil {
		gltest.Xerbla2("zchkst", err)
		return
	}

	//     Quick return if possible
	if nsizes == 0 || ntypes == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	log2ui = int(math.Log(ulpinv) / math.Log(two))
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)

	//     Loop over sizes, types
	for i = 1; i <= 4; i++ {
		iseed2[i-1] = iseed[i-1]
	}
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		if n > 0 {
			lgn = int(math.Log(float64(n)) / math.Log(two))
			if pow(2, lgn) < n {
				lgn = lgn + 1
			}
			if pow(2, lgn) < n {
				lgn = lgn + 1
			}
			lwedc = 1 + 4*n + 2*n*lgn + 4*pow(n, 2)
			lrwedc = 1 + 3*n + 2*n*lgn + 4*pow(n, 2)
			liwedc = 6 + 6*n + 5*n*lgn
		} else {
			lwedc = 8
			lrwedc = 7
			liwedc = 12
		}
		nap = (n * (n + 1)) / 2
		aninv = one / float64(max(1, n))
		//
		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}
		//
		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label300
			}
			nmats = nmats + 1
			ntest = 0
			//
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
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

			golapack.Zlaset(Full, a.Rows, n, czero, czero, a)
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
				err = matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, 0, 0, 'N', a, work)

			} else if itype == 5 {
				//              Hermitian, eigenvalues specified
				err = matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, n, n, 'N', a, work)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				err = matgen.Zlatmr(n, n, 'S', &iseed, 'H', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Hermitian, random eigenvalues
				err = matgen.Zlatmr(n, n, 'S', &iseed, 'H', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              Positive definite, eigenvalues specified.
				err = matgen.Zlatms(n, n, 'S', &iseed, 'P', rwork, imode, cond, anorm, n, n, 'N', a, work)

			} else if itype == 10 {
				//              Positive definite tridiagonal, eigenvalues specified.
				err = matgen.Zlatms(n, n, 'S', &iseed, 'P', rwork, imode, cond, anorm, 1, 1, 'N', a, work)
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

			if iinfo != 0 || err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label100:
			;

			//           Call Zhetrd and Zungtr to compute S and U from
			//           upper triangle.
			golapack.Zlacpy(Upper, n, n, a, v)

			ntest = 1
			if err = golapack.Zhetrd(Upper, n, v, sd, se, tau, work, lwork); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhetrd(U)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(0, ulpinv)
					goto label280
				}
			}

			golapack.Zlacpy(Upper, n, n, v, u)

			ntest = 2
			if err = golapack.Zungtr(Upper, n, u, tau, work, lwork); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zungtr(U)", iinfo, n, jtype, ioldsd)
				if err != nil {
					return
				} else {
					result.Set(1, ulpinv)
					goto label280
				}
			}

			//           Do tests 1 and 2
			zhet21(2, Upper, n, 1, a, sd, se, u, v, tau, work, rwork, result)
			zhet21(3, Upper, n, 1, a, sd, se, u, v, tau, work, rwork, result.Off(1))

			//           Call Zhetrd and Zungtr to compute S and U from
			//           lower triangle, do tests.
			golapack.Zlacpy(Lower, n, n, a, v)

			ntest = 3
			if err = golapack.Zhetrd(Lower, n, v, sd, se, tau, work, lwork); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhetrd(L)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(2, ulpinv)
					goto label280
				}
			}

			golapack.Zlacpy(Lower, n, n, v, u)

			ntest = 4
			if err = golapack.Zungtr(Lower, n, u, tau, work, lwork); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zungtr(L)", iinfo, n, jtype, ioldsd)
				if err != nil {
					return
				} else {
					result.Set(3, ulpinv)
					goto label280
				}
			}

			zhet21(2, Lower, n, 1, a, sd, se, u, v, tau, work, rwork, result.Off(2))
			zhet21(3, Lower, n, 1, a, sd, se, u, v, tau, work, rwork, result.Off(3))

			//           Store the upper triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = 1; jr <= jc; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call Zhptrd and Zupgtr to compute S and U from AP
			goblas.Zcopy(nap, ap.Off(0, 1), vp.Off(0, 1))

			ntest = 5
			if err = golapack.Zhptrd(Upper, n, vp, sd, se, tau); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhptrd(U)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(4, ulpinv)
					goto label280
				}
			}

			ntest = 6
			if err = golapack.Zupgtr(Upper, n, vp, tau, u, work); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zupgtr(U)", iinfo, n, jtype, ioldsd)
				if err != nil {
					return
				} else {
					result.Set(5, ulpinv)
					goto label280
				}
			}

			//           Do tests 5 and 6
			zhpt21(2, Upper, n, 1, ap, sd, se, u, vp, tau, work, rwork, result.Off(4))
			zhpt21(3, Upper, n, 1, ap, sd, se, u, vp, tau, work, rwork, result.Off(5))

			//           Store the lower triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = jc; jr <= n; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call Zhptrd and Zupgtr to compute S and U from AP
			goblas.Zcopy(nap, ap.Off(0, 1), vp.Off(0, 1))
			//
			ntest = 7
			if err = golapack.Zhptrd(Lower, n, vp, sd, se, tau); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhptrd(L)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(6, ulpinv)
					goto label280
				}
			}

			ntest = 8
			if err = golapack.Zupgtr(Lower, n, vp, tau, u, work); err != nil {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zupgtr(L)", iinfo, n, jtype, ioldsd)
				if err != nil {
					return
				} else {
					result.Set(7, ulpinv)
					goto label280
				}
			}

			zhpt21(2, Lower, n, 1, ap, sd, se, u, vp, tau, work, rwork, result.Off(6))
			zhpt21(3, Lower, n, 1, ap, sd, se, u, vp, tau, work, rwork, result.Off(7))

			//           Call Zsteqr to compute D1, D2, and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
			}
			golapack.Zlaset(Full, n, n, czero, cone, z)

			ntest = 9
			if iinfo, err = golapack.Zsteqr('V', n, d1, rwork, z, rwork.Off(n)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zsteqr(V)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
				goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
			}

			ntest = 11
			if iinfo, err = golapack.Zsteqr('N', n, d2, rwork, work.CMatrix(u.Rows, opts), rwork.Off(n)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zsteqr(N)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
				goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
			}

			ntest = 12
			if iinfo, err = golapack.Dsterf(n, d3, rwork); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsterf", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(11, ulpinv)
					goto label280
				}
			}

			//           Do Tests 9 and 10
			zstt21(n, 0, sd, se, d1, dumma, z, work, rwork, result.Off(8))

			//           Do Tests 11 and 12
			temp1 = zero
			temp2 = zero
			temp3 = zero
			temp4 = zero

			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d2.GetMag(j-1)))
				temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
				temp3 = math.Max(temp3, math.Max(d1.GetMag(j-1), d3.GetMag(j-1)))
				temp4 = math.Max(temp4, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
			}

			result.Set(10, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			result.Set(11, temp4/math.Max(unfl, ulp*math.Max(temp3, temp4)))

			//           Do Test 13 -- Sturm Sequence Test of Eigenvalues
			//                         Go up by factors of two until it succeeds
			ntest = 13
			temp1 = thresh * (half - ulp)

			for j = 0; j <= log2ui; j++ {
				iinfo = dstech(n, sd, se, d1, temp1, rwork)
				if iinfo == 0 {
					goto label170
				}
				temp1 = temp1 * two
			}

		label170:
			;
			result.Set(12, temp1)

			//           For positive definite matrices ( jtype.GT.15 ) call Zpteqr
			//           and do tests 14, 15, and 16 .
			if jtype > 15 {
				//              Compute D4 and Z4
				goblas.Dcopy(n, sd.Off(0, 1), d4.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
				}
				golapack.Zlaset(Full, n, n, czero, cone, z)

				ntest = 14
				if iinfo, err = golapack.Zpteqr('V', n, d4, rwork, z, rwork.Off(n)); err != nil || iinfo != 0 {
					fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zpteqr(V)", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if iinfo < 0 {
						return
					} else {
						result.Set(13, ulpinv)
						goto label280
					}
				}

				//              Do Tests 14 and 15
				zstt21(n, 0, sd, se, d4, dumma, z, work, rwork, result.Off(13))

				//              Compute D5
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
				}

				ntest = 16
				if iinfo, err = golapack.Zpteqr('N', n, d5, rwork, z, rwork.Off(n)); err != nil || iinfo != 0 {
					fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zpteqr(N)", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
					temp1 = math.Max(temp1, math.Max(d4.GetMag(j-1), d5.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(d4.Get(j-1)-d5.Get(j-1)))
				}

				result.Set(15, temp2/math.Max(unfl, hun*ulp*math.Max(temp1, temp2)))
			} else {
				result.Set(13, zero)
				result.Set(14, zero)
				result.Set(15, zero)
			}

			//           Call Dstebz with different options and do tests 17-18.
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
				if m, _, iinfo, err = golapack.Dstebz('A', 'E', n, vl, vu, il, iu, abstol, sd, se, wr, &iwork, toSlice(&iwork, n), rwork, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(A,rel)", iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j))/(abstol+d4.GetMag(j-1)))
				}

				result.Set(16, temp1/temp2)
			} else {
				result.Set(16, zero)
			}

			//           Now ask for all eigenvalues with high absolute accuracy.
			ntest = 18
			abstol = unfl + unfl
			if m, _, iinfo, err = golapack.Dstebz('A', 'E', n, vl, vu, il, iu, abstol, sd, se, wa1, &iwork, toSlice(&iwork, n), rwork, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(A)", iinfo, n, jtype, ioldsd)
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
				temp1 = math.Max(temp1, math.Max(d3.GetMag(j-1), wa1.GetMag(j-1)))
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
				il = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
				iu = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
				if iu < il {
					itemp = iu
					iu = il
					il = itemp
				}
			}

			if m2, _, iinfo, err = golapack.Dstebz('I', 'E', n, vl, vu, il, iu, abstol, sd, se, wa2, &iwork, toSlice(&iwork, n), rwork, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(I)", iinfo, n, jtype, ioldsd)
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

			if m3, _, iinfo, err = golapack.Dstebz('V', 'E', n, vl, vu, il, iu, abstol, sd, se, wa3, &iwork, toSlice(&iwork, n), rwork, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(V)", iinfo, n, jtype, ioldsd)
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
			temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
			temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
			if n > 0 {
				temp3 = math.Max(wa1.GetMag(n-1), wa1.GetMag(0))
			} else {
				temp3 = zero
			}

			result.Set(18, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			//           Call ZSTEIN to compute eigenvectors corresponding to
			//           eigenvalues in WA1.  (First call Dstebz again, to make sure
			//           it returns these eigenvalues in the correct order.)
			ntest = 21
			if m, _, iinfo, err = golapack.Dstebz('A', 'B', n, vl, vu, il, iu, abstol, sd, se, wa1, &iwork, toSlice(&iwork, n), rwork, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(A,B)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			if iinfo, err = golapack.Zstein(n, sd, se, m, wa1, &iwork, toSlice(&iwork, n), z, rwork, toSlice(&iwork, 2*n), toSlice(&iwork, 3*n)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "ZSTEIN", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			//           Do tests 20 and 21
			zstt21(n, 0, sd, se, wa1, dumma, z, work, rwork, result.Off(19))

			//           Call Zstedc(I) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			inde = 1
			indrwk = inde + n
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(inde-1, 1))
			}
			golapack.Zlaset(Full, n, n, czero, cone, z)

			ntest = 22
			if iinfo, err = golapack.Zstedc('I', n, d1, rwork.Off(inde-1), z, work, lwedc, rwork.Off(indrwk-1), lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstedc(I)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(21, ulpinv)
					goto label280
				}
			}

			//           Do Tests 22 and 23
			zstt21(n, 0, sd, se, d1, dumma, z, work, rwork, result.Off(21))

			//           Call Zstedc(V) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(inde-1, 1))
			}
			golapack.Zlaset(Full, n, n, czero, cone, z)

			ntest = 24
			if iinfo, err = golapack.Zstedc('V', n, d1, rwork.Off(inde-1), z, work, lwedc, rwork.Off(indrwk-1), lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstedc(V)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(23, ulpinv)
					goto label280
				}
			}

			//           Do Tests 24 and 25
			zstt21(n, 0, sd, se, d1, dumma, z, work, rwork, result.Off(23))

			//           Call Zstedc(N) to compute D2, do tests.
			//
			//           Compute D2
			goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(inde-1, 1))
			}
			golapack.Zlaset(Full, n, n, czero, cone, z)

			ntest = 26
			if iinfo, err = golapack.Zstedc('N', n, d2, rwork.Off(inde-1), z, work, lwedc, rwork.Off(indrwk-1), lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
				fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstedc(N)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
				temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d2.GetMag(j-1)))
				temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
			}

			result.Set(25, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			//           Only test Zstemr if IEEE compliant
			if ilaenv(10, "Zstemr", []byte("VA"), 1, 0, 0, 0) == 1 && ilaenv(11, "Zstemr", []byte("VA"), 1, 0, 0, 0) == 1 {
				//           Call Zstemr, do test 27 (relative eigenvalue accuracy)
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
					if m, tryrac, iinfo, err = golapack.Zstemr('V', 'A', n, sd, se, vl, vu, il, iu, wr, z, n, &iwork, tryrac, rwork, lrwork, toSlice(&iwork, 2*n), lwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(V,A,rel)", iinfo, n, jtype, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
						temp1 = math.Max(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j))/(abstol+d4.GetMag(j-1)))
					}

					result.Set(26, temp1/temp2)

					il = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}

					if crange {
						ntest = 28
						abstol = unfl + unfl
						if m, tryrac, iinfo, err = golapack.Zstemr('V', 'I', n, sd, se, vl, vu, il, iu, wr, z, n, &iwork, tryrac, rwork, lrwork, toSlice(&iwork, 2*n), lwork-2*n); err != nil || iinfo != 0 {
							fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(V,I,rel)", iinfo, n, jtype, ioldsd)
							err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
							temp1 = math.Max(temp1, math.Abs(wr.Get(j-il)-d4.Get(n-j))/(abstol+wr.GetMag(j-il)))
						}

						result.Set(27, temp1/temp2)
					} else {
						result.Set(27, zero)
					}
				} else {
					result.Set(26, zero)
					result.Set(27, zero)
				}

				//           Call Zstemr(V,I) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
				}
				golapack.Zlaset(Full, n, n, czero, cone, z)

				if crange {
					ntest = 29
					il = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}
					if m, tryrac, iinfo, err = golapack.Zstemr('V', 'I', n, d5, rwork, vl, vu, il, iu, d1, z, n, &iwork, tryrac, rwork.Off(n), lrwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(V,I)", iinfo, n, jtype, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
					//           Call Zstemr to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
					}

					ntest = 31
					if m, tryrac, iinfo, err = golapack.Zstemr('N', 'I', n, d5, rwork, vl, vu, il, iu, d2, z, n, &iwork, tryrac, rwork.Off(n), lrwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(N,I)", iinfo, n, jtype, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
						temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d2.GetMag(j-1)))
						temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(30, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

					//           Call Zstemr(V,V) to compute D1 and Z, do tests.
					//
					//           Compute D1 and Z
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
					}
					golapack.Zlaset(Full, n, n, czero, cone, z)

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

					if m, tryrac, iinfo, err = golapack.Zstemr('V', 'V', n, d5, rwork, vl, vu, il, iu, d1, z, m, &iwork, tryrac, rwork.Off(n), lrwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(V,V)", iinfo, n, jtype, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						if iinfo < 0 {
							return
						} else {
							result.Set(31, ulpinv)
							goto label280
						}
					}

					//           Do Tests 32 and 33
					zstt22(n, m, 0, sd, se, d1, dumma, z, work.CMatrix(m, opts), rwork, result.Off(31))

					//           Call Zstemr to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
					}

					ntest = 34
					if m, tryrac, iinfo, err = golapack.Zstemr('N', 'V', n, d5, rwork, vl, vu, il, iu, d2, z, n, &iwork, tryrac, rwork.Off(n), lrwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(N,V)", iinfo, n, jtype, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
						temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d2.GetMag(j-1)))
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

				//           Call Zstemr(V,A) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
				}

				ntest = 35

				if m, tryrac, iinfo, err = golapack.Zstemr('V', 'A', n, d5, rwork, vl, vu, il, iu, d1, z, n, &iwork, tryrac, rwork.Off(n), lrwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(V,A)", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if iinfo < 0 {
						return
					} else {
						result.Set(34, ulpinv)
						goto label280
					}
				}

				//           Do Tests 35 and 36
				zstt22(n, m, 0, sd, se, d1, dumma, z, work.CMatrix(m, opts), rwork, result.Off(34))

				//           Call Zstemr to compute D2, do tests.
				//
				//           Compute D2
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), rwork.Off(0, 1))
				}

				ntest = 37
				if m, tryrac, iinfo, err = golapack.Zstemr('N', 'A', n, d5, rwork, vl, vu, il, iu, d2, z, n, &iwork, tryrac, rwork.Off(n), lrwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zchkst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zstemr(N,A)", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
					temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d2.GetMag(j-1)))
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
				if result.Get(jr-1) >= thresh {
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Complex Hermitian eigenvalue problem\n", "ZST")
						fmt.Printf(" Matrix types (see zchkst for details): \n")
						fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
						fmt.Printf(" Dense %s Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Hermitian")
						fmt.Printf(" 16=Positive definite, evenly spaced eigenvalues\n 17=Positive definite, geometrically spaced eigenvlaues\n 18=Positive definite, clustered eigenvalues\n 19=Positive definite, small evenly spaced eigenvalues\n 20=Positive definite, large evenly spaced eigenvalues\n 21=Diagonally dominant tridiagonal, geometrically spaced eigenvalues\n")

						//                    Tests performed
						fmt.Printf("\nTest performed:  see zchkst for details.\n\n")
					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
						err = fmt.Errorf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
						err = fmt.Errorf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}
		label300:
		}
	}

	//     Summary
	dlasum("Zst", nerrs, ntestt)

	return
}
