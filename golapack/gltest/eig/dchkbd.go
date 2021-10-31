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

// dchkbd checks the singular value decomposition (SVD) routines.
//
// Dgebrd reduces a real general m by n matrix A to upper or lower
// bidiagonal form B by an orthogonal transformation:  Q' * A * P = B
// (or A = Q * B * P').  The matrix B is upper bidiagonal if m >= n
// and lower bidiagonal if m < n.
//
// Dorgbr generates the orthogonal matrices Q and P' from Dgebrd.
// Note that Q and P are not necessarily square.
//
// Dbdsqr computes the singular value decomposition of the bidiagonal
// matrix B as B = U S V'.  It is called three times to compute
//    1)  B = U S1 V', where S1 is the diagonal matrix of singular
//        values and the columns of the matrices U and V are the left
//        and right singular vectors, respectively, of B.
//    2)  Same as 1), but the singular values are stored in S2 and the
//        singular vectors are not computed.
//    3)  A = (UQ) S (P'V'), the SVD of the original matrix A.
// In addition, Dbdsqr has an option to apply the left orthogonal matrix
// U to a matrix X, useful in least squares applications.
//
// Dbdsdc computes the singular value decomposition of the bidiagonal
// matrix B as B = U S V' using divide-and-conquer. It is called twice
// to compute
//    1) B = U S1 V', where S1 is the diagonal matrix of singular
//        values and the columns of the matrices U and V are the left
//        and right singular vectors, respectively, of B.
//    2) Same as 1), but the singular values are stored in S2 and the
//        singular vectors are not computed.
//
//  Dbdsvdx computes the singular value decomposition of the bidiagonal
//  matrix B as B = U S V' using bisection and inverse iteration. It is
//  called six times to compute
//     1) B = U S1 V', RANGE='A', where S1 is the diagonal matrix of singular
//         values and the columns of the matrices U and V are the left
//         and right singular vectors, respectively, of B.
//     2) Same as 1), but the singular values are stored in S2 and the
//         singular vectors are not computed.
//     3) B = U S1 V', RANGE='I', with where S1 is the diagonal matrix of singular
//         values and the columns of the matrices U and V are the left
//         and right singular vectors, respectively, of B
//     4) Same as 3), but the singular values are stored in S2 and the
//         singular vectors are not computed.
//     5) B = U S1 V', RANGE='V', with where S1 is the diagonal matrix of singular
//         values and the columns of the matrices U and V are the left
//         and right singular vectors, respectively, of B
//     6) Same as 5), but the singular values are stored in S2 and the
//         singular vectors are not computed.
//
// For each pair of matrix dimensions (M,N) and each selected matrix
// type, an M by N matrix A and an M by NRHS matrix X are generated.
// The problem dimensions are as follows
//    A:          M x N
//    Q:          M x min(M,N) (but M x M if NRHS > 0)
//    P:          min(M,N) x N
//    B:          min(M,N) x min(M,N)
//    U, V:       min(M,N) x min(M,N)
//    S1, S2      diagonal, order min(M,N)
//    X:          M x NRHS
//
// For each generated matrix, 14 tests are performed:
//
// Test Dgebrd and Dorgbr
//
// (1)   | A - Q B PT | / ( |A| max(M,N) ulp ), PT = P'
//
// (2)   | I - Q' Q | / ( M ulp )
//
// (3)   | I - PT PT' | / ( N ulp )
//
// Test Dbdsqr on bidiagonal matrix B
//
// (4)   | B - U S1 VT | / ( |B| min(M,N) ulp ), VT = V'
//
// (5)   | Y - U Z | / ( |Y| max(min(M,N),k) ulp ), where Y = Q' X
//                                                  and   Z = U' Y.
// (6)   | I - U' U | / ( min(M,N) ulp )
//
// (7)   | I - VT VT' | / ( min(M,N) ulp )
//
// (8)   S1 contains min(M,N) nonnegative values in decreasing order.
//       (Return 0 if true, 1/ULP if false.)
//
// (9)   | S1 - S2 | / ( |S1| ulp ), where S2 is computed without
//                                   computing U and V.
//
// (10)  0 if the true singular values of B are within THRESH of
//       those in S1.  2*THRESH if they are not.  (Tested using
//       DSVDCH)
//
// Test Dbdsqr on matrix A
//
// (11)  | A - (QU) S (VT PT) | / ( |A| max(M,N) ulp )
//
// (12)  | X - (QU) Z | / ( |X| max(M,k) ulp )
//
// (13)  | I - (QU)'(QU) | / ( M ulp )
//
// (14)  | I - (VT PT) (PT'VT') | / ( N ulp )
//
// Test Dbdsdc on bidiagonal matrix B
//
// (15)  | B - U S1 VT | / ( |B| min(M,N) ulp ), VT = V'
//
// (16)  | I - U' U | / ( min(M,N) ulp )
//
// (17)  | I - VT VT' | / ( min(M,N) ulp )
//
// (18)  S1 contains min(M,N) nonnegative values in decreasing order.
//       (Return 0 if true, 1/ULP if false.)
//
// (19)  | S1 - S2 | / ( |S1| ulp ), where S2 is computed without
//                                   computing U and V.
//  Test Dbdsvdx on bidiagonal matrix B
//
//  (20)  | B - U S1 VT | / ( |B| min(M,N) ulp ), VT = V'
//
//  (21)  | I - U' U | / ( min(M,N) ulp )
//
//  (22)  | I - VT VT' | / ( min(M,N) ulp )
//
//  (23)  S1 contains min(M,N) nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
//  (24)  | S1 - S2 | / ( |S1| ulp ), where S2 is computed without
//                                    computing U and V.
//
//  (25)  | S1 - U' B VT' | / ( |S| n ulp )    Dbdsvdx('V', 'I')
//
//  (26)  | I - U' U | / ( min(M,N) ulp )
//
//  (27)  | I - VT VT' | / ( min(M,N) ulp )
//
//  (28)  S1 contains min(M,N) nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
//  (29)  | S1 - S2 | / ( |S1| ulp ), where S2 is computed without
//                                    computing U and V.
//
//  (30)  | S1 - U' B VT' | / ( |S1| n ulp )   Dbdsvdx('V', 'V')
//
//  (31)  | I - U' U | / ( min(M,N) ulp )
//
//  (32)  | I - VT VT' | / ( min(M,N) ulp )
//
//  (33)  S1 contains min(M,N) nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
//  (34)  | S1 - S2 | / ( |S1| ulp ), where S2 is computed without
//                                    computing U and V.
//
// The possible matrix types are
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
// (6)  Same as (3), but multiplied by math.Sqrt( overflow threshold )
// (7)  Same as (3), but multiplied by math.Sqrt( underflow threshold )
//
// (8)  A matrix of the form  U D V, where U and V are orthogonal and
//      D has evenly spaced entries 1, ..., ULP with random signs
//      on the diagonal.
//
// (9)  A matrix of the form  U D V, where U and V are orthogonal and
//      D has geometrically spaced entries 1, ..., ULP with random
//      signs on the diagonal.
//
// (10) A matrix of the form  U D V, where U and V are orthogonal and
//      D has "clustered" entries 1, ULP,..., ULP with random
//      signs on the diagonal.
//
// (11) Same as (8), but multiplied by math.Sqrt( overflow threshold )
// (12) Same as (8), but multiplied by math.Sqrt( underflow threshold )
//
// (13) Rectangular matrix with random entries chosen from (-1,1).
// (14) Same as (13), but multiplied by math.Sqrt( overflow threshold )
// (15) Same as (13), but multiplied by math.Sqrt( underflow threshold )
//
// Special case:
// (16) A bidiagonal matrix with random entries chosen from a
//      logarithmic distribution on [ulp^2,ulp^(-2)]  (I.e., each
//      entry is  e^x, where x is chosen uniformly on
//      [ 2 math.Log(ulp), -2 math.Log(ulp) ] .)  For *this* type:
//      (a) Dgebrd is not called to reduce it to bidiagonal form.
//      (b) the bidiagonal is  min(M,N) x min(M,N); if M<N, the
//          matrix will be lower bidiagonal, otherwise upper.
//      (c) only tests 5--8 and 14 are performed.
//
// A subset of the full set of matrix types may be selected through
// the logical array DOTYPE.
func dchkbd(nsizes int, mval []int, nval []int, ntypes int, dotype []bool, nrhs int, iseed *[]int, thresh float64, a *mat.Matrix, bd, be, s1, s2 *mat.Vector, x, y, z, q, pt, u, vt *mat.Matrix, work *mat.Vector, lwork int, iwork []int, nout int, t *testing.T) (nfail, ntest int, err error) {
	var badmm, badnn, bidiag bool
	var uplo mat.MatUplo
	var amninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, il, imode, itemp, itype, iu, iwbd, iwbe, iwbs, iwbz, iwwork, j, jcol, jsize, jtype, log2ui, m, maxtyp, minwrk, mmax, mnmax, mnmin, mnmin2, mq, mtypes, n, nmax, ns1, ns2 int

	dum := vf(1)
	dumma := vf(1)
	result := vf(40)
	idum := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	kmagn := make([]int, 16)
	kmode := make([]int, 16)
	ktype := make([]int, 16)

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	maxtyp = 16

	infot := &gltest.Common.Infoc.Infot

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15] = 1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9, 10
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 0
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 0

	//     Check for errors

	badmm = false
	badnn = false
	mmax = 1
	nmax = 1
	mnmax = 1
	minwrk = 1
	for j = 1; j <= nsizes; j++ {
		mmax = max(mmax, mval[j-1])
		if mval[j-1] < 0 {
			badmm = true
		}
		nmax = max(nmax, nval[j-1])
		if nval[j-1] < 0 {
			badnn = true
		}
		mnmax = max(mnmax, min(mval[j-1], nval[j-1]))
		minwrk = max(minwrk, 3*(mval[j-1]+nval[j-1]), mval[j-1]*(mval[j-1]+max(mval[j-1], nval[j-1], nrhs)+1)+nval[j-1]*min(nval[j-1], mval[j-1]))
	}

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsizes=%v", nsizes)
	} else if badmm {
		err = fmt.Errorf("badmm: mval=%v", mval)
	} else if badnn {
		err = fmt.Errorf("badnn, nval=%v", nval)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < mmax {
		err = fmt.Errorf("a.Rows < mmax: a.Rows=%v, mmax=%v", a.Rows, mmax)
	} else if x.Rows < mmax {
		err = fmt.Errorf("x.Rows < mmax: x.Rows=%v, mmax=%v", x.Rows, mmax)
	} else if q.Rows < mmax {
		err = fmt.Errorf("q.Rows < mmax: q.Rows=%v, mmax=%v", q.Rows, mmax)
	} else if pt.Rows < mnmax {
		err = fmt.Errorf("pt.Rows < mmax: pt.Rows=%v, mmax=%v", pt.Rows, mmax)
	} else if minwrk > lwork {
		err = fmt.Errorf("minwrk > lwork: minwrk=%v, lwork=%v", minwrk, lwork)
	}

	if err != nil {
		t.Fail()
		gltest.Xerbla2("dchkbd", err)
		return
	}

	//     Initialize constants
	path := "Dbd"
	nfail = 0
	ntest = 0
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	log2ui = int(math.Log(ulpinv) / math.Log(two))
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)
	(*infot) = 0

	//     Loop over sizes, types
	for jsize = 1; jsize <= nsizes; jsize++ {
		m = mval[jsize-1]
		n = nval[jsize-1]
		mnmin = min(m, n)
		amninv = one / float64(max(m, n, 1))

		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label290
			}

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			for j = 1; j <= 34; j++ {
				result.Set(j-1, -one)
			}

			uplo = Full

			//           Compute "A"
			//
			//           Control parameters:
			//
			//           KMAGN  KMODE        KTYPE
			//       =1  O(1)   clustered 1  zero
			//       =2  large  clustered 2  identity
			//       =3  small  exponential  (none)
			//       =4         arithmetic   diagonal, (w/ eigenvalues)
			//       =5         random       symmetric, w/ eigenvalues
			//       =6                      nonsymmetric, w/ singular values
			//       =7                      random diagonal
			//       =8                      random symmetric
			//       =9                      random nonsymmetric
			//       =10                     random bidiagonal (math.Log. distrib.)
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
			anorm = (rtovfl * ulp) * amninv
			goto label70

		label60:
			;
			anorm = rtunfl * float64(max(m, n)) * ulpinv
			goto label70

		label70:
			;

			golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
			iinfo = 0
			cond = ulpinv

			bidiag = false
			if itype == 1 {
				//              Zero matrix
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jcol = 1; jcol <= mnmin; jcol++ {
					a.Set(jcol-1, jcol-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				iinfo, err = matgen.Dlatms(mnmin, mnmin, 'S', iseed, 'N', work, imode, cond, anorm, 0, 0, 'N', a, work.Off(mnmin))

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				iinfo, err = matgen.Dlatms(mnmin, mnmin, 'S', iseed, 'S', work, imode, cond, anorm, m, n, 'N', a, work.Off(mnmin))

			} else if itype == 6 {
				//              Nonsymmetric, singular values specified
				iinfo, err = matgen.Dlatms(m, n, 'S', iseed, 'N', work, imode, cond, anorm, m, n, 'N', a, work.Off(mnmin))

			} else if itype == 7 {
				//              Diagonal, random entries
				iinfo, err = matgen.Dlatmr(mnmin, mnmin, 'S', iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(mnmin), 1, one, work.Off(2*mnmin), 1, one, 'N', &iwork, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Symmetric, random entries
				iinfo, err = matgen.Dlatmr(mnmin, mnmin, 'S', iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(mnmin), 1, one, work.Off(m+mnmin), 1, one, 'N', &iwork, m, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              Nonsymmetric, random entries
				iinfo, err = matgen.Dlatmr(m, n, 'S', iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(mnmin), 1, one, work.Off(m+mnmin), 1, one, 'N', &iwork, m, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 10 {
				//              Bidiagonal, random entries
				temp1 = -two * math.Log(ulp)
				for j = 1; j <= mnmin; j++ {
					bd.Set(j-1, math.Exp(temp1*matgen.Dlarnd(2, iseed)))
					if j < mnmin {
						be.Set(j-1, math.Exp(temp1*matgen.Dlarnd(2, iseed)))
					}
				}

				iinfo = 0
				bidiag = true
				if m >= n {
					uplo = Upper
				} else {
					uplo = Lower
				}
			} else {
				iinfo = 1
			}

			if iinfo == 0 {
				//              Generate Right-Hand Side
				if bidiag {
					iinfo, err = matgen.Dlatmr(mnmin, nrhs, 'S', iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(mnmin), 1, one, work.Off(2*mnmin), 1, one, 'N', &iwork, mnmin, nrhs, zero, one, 'N', y, &iwork)
				} else {
					iinfo, err = matgen.Dlatmr(m, nrhs, 'S', iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(m), 1, one, work.Off(2*m), 1, one, 'N', &iwork, m, nrhs, zero, one, 'N', x, &iwork)
				}
			}

			//           Error Exit
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label100:
			;

			//           Call Dgebrd and Dorgbr to compute B, Q, and P, do tests.
			if !bidiag {
				//              Compute transformations to reduce A to bidiagonal form:
				//              B := Q' * A * P.
				golapack.Dlacpy(Full, m, n, a, q)
				if err = golapack.Dgebrd(m, n, q, bd, be, work, work.Off(mnmin), work.Off(2*mnmin), lwork-2*mnmin); err != nil {
					t.Fail()
					fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dgebrd", iinfo, m, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				golapack.Dlacpy(Full, m, n, q, pt)
				if m >= n {
					uplo = Upper
				} else {
					uplo = Lower
				}

				//              Generate Q
				mq = m
				if nrhs <= 0 {
					mq = mnmin
				}
				if err = golapack.Dorgbr('Q', m, mq, n, q, work, work.Off(2*mnmin), lwork-2*mnmin); err != nil {
					t.Fail()
					fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dorgbr(Q)", iinfo, m, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Generate P'
				if err = golapack.Dorgbr('P', mnmin, n, m, pt, work.Off(mnmin), work.Off(2*mnmin), lwork-2*mnmin); err != nil {
					t.Fail()
					fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dorgbr(P)", iinfo, m, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Apply Q' to an M by NRHS matrix X:  Y := Q' * X.
				if err = goblas.Dgemm(Trans, NoTrans, m, nrhs, m, one, q, x, zero, y); err != nil {
					panic(err)
				}

				//              Test 1:  Check the decomposition A := Q * B * PT
				//                   2:  Check the orthogonality of Q
				//                   3:  Check the orthogonality of PT
				result.Set(0, dbdt01(m, n, 1, a, q, bd, be, pt, work))
				result.Set(1, dort01('C', m, mq, q, work, lwork))
				result.Set(2, dort01('R', mnmin, n, pt, work, lwork))
			}

			//           Use Dbdsqr to form the SVD of the bidiagonal matrix B:
			//           B := U * S1 * VT, and compute Z = U' * Y.
			goblas.Dcopy(mnmin, bd.Off(0, 1), s1.Off(0, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlacpy(Full, m, nrhs, y, z)
			golapack.Dlaset(Full, mnmin, mnmin, zero, one, u)
			golapack.Dlaset(Full, mnmin, mnmin, zero, one, vt)

			iinfo, err = golapack.Dbdsqr(uplo, mnmin, mnmin, mnmin, nrhs, s1, work, vt, u, z, work.Off(mnmin))

			//           Check error code from Dbdsqr.
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsqr(vects)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(3, ulpinv)
					goto label270
				}
			}

			//           Use Dbdsqr to compute only the singular values of the
			//           bidiagonal matrix B;  U, VT, and Z should not be modified.
			goblas.Dcopy(mnmin, bd.Off(0, 1), s2.Off(0, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(0, 1))
			}

			iinfo, err = golapack.Dbdsqr(uplo, mnmin, 0, 0, 0, s2, work, vt, u, z, work.Off(mnmin))

			//           Check error code from Dbdsqr.
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsqr(values)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(8, ulpinv)
					goto label270
				}
			}

			//           Test 4:  Check the decomposition B := U * S1 * VT
			//                5:  Check the computation Z := U' * Y
			//                6:  Check the orthogonality of U
			//                7:  Check the orthogonality of VT
			result.Set(3, dbdt03(uplo, mnmin, 1, bd, be, u, s1, vt, work))
			result.Set(4, dbdt02(mnmin, nrhs, y, z, u, work))
			result.Set(5, dort01('C', mnmin, mnmin, u, work, lwork))
			result.Set(6, dort01('R', mnmin, mnmin, vt, work, lwork))

			//           Test 8:  Check that the singular values are sorted in
			//                    non-increasing order and are non-negative
			result.Set(7, zero)
			for i = 1; i <= mnmin-1; i++ {
				if s1.Get(i-1) < s1.Get(i) {
					result.Set(7, ulpinv)
				}
				if s1.Get(i-1) < zero {
					result.Set(7, ulpinv)
				}
			}
			if mnmin >= 1 {
				if s1.Get(mnmin-1) < zero {
					result.Set(7, ulpinv)
				}
			}

			//           Test 9:  Compare Dbdsqr with and without singular vectors
			temp2 = zero

			for j = 1; j <= mnmin; j++ {
				temp1 = math.Abs(s1.Get(j-1)-s2.Get(j-1)) / math.Max(math.Sqrt(unfl)*math.Max(s1.Get(0), one), ulp*math.Max(math.Abs(s1.Get(j-1)), math.Abs(s2.Get(j-1))))
				temp2 = math.Max(temp1, temp2)
			}

			result.Set(8, temp2)

			//           Test 10:  Sturm sequence test of singular values
			//                     Go up by factors of two until it succeeds
			temp1 = thresh * (half - ulp)

			for j = 0; j <= log2ui; j++ {
				//               CALL DSVDCH( MNMIN, BD, BE, S1, TEMP1, IINFO )
				if iinfo == 0 {
					goto label140
				}
				temp1 = temp1 * two
			}

		label140:
			;
			result.Set(9, temp1)

			//           Use Dbdsqr to form the decomposition A := (QU) S (VT PT)
			//           from the bidiagonal form A := Q B PT.
			if !bidiag {
				goblas.Dcopy(mnmin, bd.Off(0, 1), s2.Off(0, 1))
				if mnmin > 0 {
					goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(0, 1))
				}

				iinfo, err = golapack.Dbdsqr(uplo, mnmin, n, m, nrhs, s2, work, pt, q, y, work.Off(mnmin))

				//              Test 11:  Check the decomposition A := Q*U * S2 * VT*PT
				//                   12:  Check the computation Z := U' * Q' * X
				//                   13:  Check the orthogonality of Q*U
				//                   14:  Check the orthogonality of VT*PT
				result.Set(10, dbdt01(m, n, 0, a, q, s2, dumma, pt, work))
				result.Set(11, dbdt02(m, nrhs, x, y, q, work))
				result.Set(12, dort01('C', m, mq, q, work, lwork))
				result.Set(13, dort01('R', mnmin, n, pt, work, lwork))
			}

			//           Use Dbdsdc to form the SVD of the bidiagonal matrix B:
			//           B := U * S1 * VT
			goblas.Dcopy(mnmin, bd.Off(0, 1), s1.Off(0, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset(Full, mnmin, mnmin, zero, one, u)
			golapack.Dlaset(Full, mnmin, mnmin, zero, one, vt)

			if err = golapack.Dbdsdc(uplo, 'I', mnmin, s1, work, u, vt, dum, &idum, work.Off(mnmin), &iwork); err != nil {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsdc(vects)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(14, ulpinv)
					goto label270
				}
			}

			//           Use Dbdsdc to compute only the singular values of the
			//           bidiagonal matrix B;  U and VT should not be modified.
			goblas.Dcopy(mnmin, bd.Off(0, 1), s2.Off(0, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(0, 1))
			}

			if err = golapack.Dbdsdc(uplo, 'N', mnmin, s2, work, dum.Matrix(1, opts), dum.Matrix(1, opts), dum, &idum, work.Off(mnmin), &iwork); err != nil {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsdc(values)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(17, ulpinv)
					goto label270
				}
			}

			//           Test 15:  Check the decomposition B := U * S1 * VT
			//                16:  Check the orthogonality of U
			//                17:  Check the orthogonality of VT
			result.Set(14, dbdt03(uplo, mnmin, 1, bd, be, u, s1, vt, work))
			result.Set(15, dort01('C', mnmin, mnmin, u, work, lwork))
			result.Set(16, dort01('R', mnmin, mnmin, vt, work, lwork))

			//           Test 18:  Check that the singular values are sorted in
			//                     non-increasing order and are non-negative
			result.Set(17, zero)
			for i = 1; i <= mnmin-1; i++ {
				if s1.Get(i-1) < s1.Get(i) {
					result.Set(17, ulpinv)
				}
				if s1.Get(i-1) < zero {
					result.Set(17, ulpinv)
				}
			}
			if mnmin >= 1 {
				if s1.Get(mnmin-1) < zero {
					result.Set(17, ulpinv)
				}
			}

			//           Test 19:  Compare Dbdsqr with and without singular vectors
			temp2 = zero

			for j = 1; j <= mnmin; j++ {
				temp1 = math.Abs(s1.Get(j-1)-s2.Get(j-1)) / math.Max(math.Sqrt(unfl)*math.Max(s1.Get(0), one), ulp*math.Max(math.Abs(s1.Get(0)), math.Abs(s2.Get(0))))
				temp2 = math.Max(temp1, temp2)
			}

			result.Set(18, temp2)

			//           Use Dbdsvdx to compute the SVD of the bidiagonal matrix B:
			//           B := U * S1 * VT
			if jtype == 10 || jtype == 16 {
				//              =================================
				//              Matrix types temporarily disabled
				//              =================================
				result.Set(19, zero)
				goto label270
			}

			iwbs = 1
			iwbd = iwbs + mnmin
			iwbe = iwbd + mnmin
			iwbz = iwbe + mnmin
			iwwork = iwbz + 2*mnmin*(mnmin+1)
			mnmin2 = max(1, mnmin*2)

			goblas.Dcopy(mnmin, bd.Off(0, 1), work.Off(iwbd-1, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(iwbe-1, 1))
			}

			iinfo, err = golapack.Dbdsvdx(uplo, 'V', 'A', mnmin, work.Off(iwbd-1), work.Off(iwbe-1), zero, zero, 0, 0, ns1, s1, work.MatrixOff(iwbz-1, mnmin2, opts), work.Off(iwwork-1), &iwork)

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsvdx(vects,A)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					goto label270
				}
			}

			j = iwbz
			for i = 1; i <= ns1; i++ {
				goblas.Dcopy(mnmin, work.Off(j-1, 1), u.Vector(0, i-1, 1))
				j = j + mnmin
				goblas.Dcopy(mnmin, work.Off(j-1, 1), vt.Vector(i-1, 0, *&pt.Rows))
				j = j + mnmin
			}

			//           Use Dbdsvdx to compute only the singular values of the
			//           bidiagonal matrix B;  U and VT should not be modified.
			if jtype == 9 {
				//              =================================
				//              Matrix types temporarily disabled
				//              =================================
				result.Set(23, zero)
				goto label270
			}

			goblas.Dcopy(mnmin, bd.Off(0, 1), work.Off(iwbd-1, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(iwbe-1, 1))
			}

			iinfo, err = golapack.Dbdsvdx(uplo, 'N', 'A', mnmin, work.Off(iwbd-1), work.Off(iwbe-1), zero, zero, 0, 0, ns2, s2, work.MatrixOff(iwbz-1, mnmin2, opts), work.Off(iwwork-1), &iwork)

			//           Check error code from Dbdsvdx.
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsvdx(values,A)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(23, ulpinv)
					goto label270
				}
			}

			//           Save S1 for tests 30-34.
			goblas.Dcopy(mnmin, s1.Off(0, 1), work.Off(iwbs-1, 1))

			//           Test 20:  Check the decomposition B := U * S1 * VT
			//                21:  Check the orthogonality of U
			//                22:  Check the orthogonality of VT
			//                23:  Check that the singular values are sorted in
			//                     non-increasing order and are non-negative
			//                24:  Compare Dbdsvdx with and without singular vectors
			result.Set(19, dbdt03(uplo, mnmin, 1, bd, be, u, s1, vt, work.Off(iwbs+mnmin-1)))
			result.Set(20, dort01('C', mnmin, mnmin, u, work.Off(iwbs+mnmin-1), lwork-mnmin))
			result.Set(21, dort01('R', mnmin, mnmin, vt, work.Off(iwbs+mnmin-1), lwork-mnmin))

			result.Set(22, zero)
			for i = 1; i <= mnmin-1; i++ {
				if s1.Get(i-1) < s1.Get(i) {
					result.Set(22, ulpinv)
				}
				if s1.Get(i-1) < zero {
					result.Set(22, ulpinv)
				}
			}
			if mnmin >= 1 {
				if s1.Get(mnmin-1) < zero {
					result.Set(22, ulpinv)
				}
			}

			temp2 = zero
			for j = 1; j <= mnmin; j++ {
				temp1 = math.Abs(s1.Get(j-1)-s2.Get(j-1)) / math.Max(math.Sqrt(unfl)*math.Max(s1.Get(0), one), ulp*math.Max(math.Abs(s1.Get(0)), math.Abs(s2.Get(0))))
				temp2 = math.Max(temp1, temp2)
			}
			result.Set(23, temp2)
			anorm = s1.Get(0)

			//           Use Dbdsvdx with RANGE='I': choose random values for IL and
			//           IU, and ask for the IL-th through IU-th singular values
			//           and corresponding vectors.
			for i = 1; i <= 4; i++ {
				iseed2[i-1] = (*iseed)[i-1]
			}
			if mnmin <= 1 {
				il = 1
				iu = mnmin
			} else {
				il = 1 + int(float64(mnmin-1)*matgen.Dlarnd(1, &iseed2))
				iu = 1 + int(float64(mnmin-1)*matgen.Dlarnd(1, &iseed2))
				if iu < il {
					itemp = iu
					iu = il
					il = itemp
				}
			}

			goblas.Dcopy(mnmin, bd.Off(0, 1), work.Off(iwbd-1, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(iwbe-1, 1))
			}

			iinfo, err = golapack.Dbdsvdx(uplo, 'V', 'I', mnmin, work.Off(iwbd-1), work.Off(iwbe-1), zero, zero, il, iu, ns1, s1, work.MatrixOff(iwbz-1, mnmin2, opts), work.Off(iwwork-1), &iwork)

			//           Check error code from Dbdsvdx.
			if iinfo != 0 {
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsvdx(vects,I)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(24, ulpinv)
					goto label270
				}
			}

			j = iwbz
			for i = 1; i <= ns1; i++ {
				goblas.Dcopy(mnmin, work.Off(j-1, 1), u.Vector(0, i-1, 1))
				j = j + mnmin
				goblas.Dcopy(mnmin, work.Off(j-1, 1), vt.Vector(i-1, 0, *&pt.Rows))
				j = j + mnmin
			}

			//           Use Dbdsvdx to compute only the singular values of the
			//           bidiagonal matrix B;  U and VT should not be modified.
			goblas.Dcopy(mnmin, bd.Off(0, 1), work.Off(iwbd-1, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(iwbe-1, 1))
			}

			iinfo, err = golapack.Dbdsvdx(uplo, 'N', 'I', mnmin, work.Off(iwbd-1), work.Off(iwbe-1), zero, zero, il, iu, ns2, s2, work.MatrixOff(iwbz-1, mnmin2, opts), work.Off(iwwork-1), &iwork)

			//           Check error code from Dbdsvdx.
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsvdx(values,I)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(28, ulpinv)
					goto label270
				}
			}

			//           Test 25:  Check S1 - U' * B * VT'
			//                26:  Check the orthogonality of U
			//                27:  Check the orthogonality of VT
			//                28:  Check that the singular values are sorted in
			//                     non-increasing order and are non-negative
			//                29:  Compare Dbdsvdx with and without singular vectors
			result.Set(24, dbdt04(uplo, mnmin, bd, be, s1, ns1, u, vt, work.Off(iwbs+mnmin-1)))
			result.Set(25, dort01('C', mnmin, ns1, u, work.Off(iwbs+mnmin-1), lwork-mnmin))
			result.Set(26, dort01('R', ns1, mnmin, vt, work.Off(iwbs+mnmin-1), lwork-mnmin))

			result.Set(27, zero)
			for i = 1; i <= ns1-1; i++ {
				if s1.Get(i-1) < s1.Get(i) {
					result.Set(27, ulpinv)
				}
				if s1.Get(i-1) < zero {
					result.Set(27, ulpinv)
				}
			}
			if ns1 >= 1 {
				if s1.Get(ns1-1) < zero {
					result.Set(27, ulpinv)
				}
			}

			temp2 = zero
			for j = 1; j <= ns1; j++ {
				temp1 = math.Abs(s1.Get(j-1)-s2.Get(j-1)) / math.Max(math.Sqrt(unfl)*math.Max(s1.Get(0), one), ulp*math.Max(math.Abs(s1.Get(0)), math.Abs(s2.Get(0))))
				temp2 = math.Max(temp1, temp2)
			}
			result.Set(28, temp2)

			//           Use Dbdsvdx with RANGE='V': determine the values VL and VU
			//           of the IL-th and IU-th singular values and ask for all
			//           singular values in this range.
			goblas.Dcopy(mnmin, work.Off(iwbs-1, 1), s1.Off(0, 1))

			if mnmin > 0 {
				if il != 1 {
					vu = s1.Get(il-1) + math.Max(half*math.Abs(s1.Get(il-1)-s1.Get(il-1-1)), math.Max(ulp*anorm, two*rtunfl))
				} else {
					vu = s1.Get(0) + math.Max(half*math.Abs(s1.Get(mnmin-1)-s1.Get(0)), math.Max(ulp*anorm, two*rtunfl))
				}
				if iu != ns1 {
					vl = s1.Get(iu-1) - math.Max(ulp*anorm, math.Max(two*rtunfl, half*math.Abs(s1.Get(iu)-s1.Get(iu-1))))
				} else {
					vl = s1.Get(ns1-1) - math.Max(ulp*anorm, math.Max(two*rtunfl, half*math.Abs(s1.Get(mnmin-1)-s1.Get(0))))
				}
				vl = math.Max(vl, zero)
				vu = math.Max(vu, zero)
				if vl >= vu {
					vu = math.Max(vu*2, vu+vl+half)
				}
			} else {
				vl = zero
				vu = one
			}

			goblas.Dcopy(mnmin, bd.Off(0, 1), work.Off(iwbd-1, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(iwbe-1, 1))
			}

			iinfo, err = golapack.Dbdsvdx(uplo, 'V', 'V', mnmin, work.Off(iwbd-1), work.Off(iwbe-1), vl, vu, 0, 0, ns1, s1, work.MatrixOff(iwbz-1, mnmin2, opts), work.Off(iwwork-1), &iwork)

			//           Check error code from Dbdsvdx.
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsvdx(vects,V)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(29, ulpinv)
					goto label270
				}
			}

			j = iwbz
			for i = 1; i <= ns1; i++ {
				goblas.Dcopy(mnmin, work.Off(j-1, 1), u.Vector(0, i-1, 1))
				j = j + mnmin
				goblas.Dcopy(mnmin, work.Off(j-1, 1), vt.Vector(i-1, 0, *&pt.Rows))
				j = j + mnmin
			}

			//           Use Dbdsvdx to compute only the singular values of the
			//           bidiagonal matrix B;  U and VT should not be modified.
			goblas.Dcopy(mnmin, bd.Off(0, 1), work.Off(iwbd-1, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), work.Off(iwbe-1, 1))
			}

			iinfo, err = golapack.Dbdsvdx(uplo, 'N', 'V', mnmin, work.Off(iwbd-1), work.Off(iwbe-1), vl, vu, 0, 0, ns2, s2, work.MatrixOff(iwbz-1, mnmin2, opts), work.Off(iwwork-1), &iwork)

			//           Check error code from Dbdsvdx.
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" dchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Dbdsvdx(values,V)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(33, ulpinv)
					goto label270
				}
			}

			//           Test 30:  Check S1 - U' * B * VT'
			//                31:  Check the orthogonality of U
			//                32:  Check the orthogonality of VT
			//                33:  Check that the singular values are sorted in
			//                     non-increasing order and are non-negative
			//                34:  Compare Dbdsvdx with and without singular vectors
			result.Set(29, dbdt04(uplo, mnmin, bd, be, s1, ns1, u, vt, work.Off(iwbs+mnmin-1)))
			result.Set(30, dort01('C', mnmin, ns1, u, work.Off(iwbs+mnmin-1), lwork-mnmin))
			result.Set(31, dort01('R', ns1, mnmin, vt, work.Off(iwbs+mnmin-1), lwork-mnmin))

			result.Set(32, zero)
			for i = 1; i <= ns1-1; i++ {
				if s1.Get(i-1) < s1.Get(i) {
					result.Set(27, ulpinv)
				}
				if s1.Get(i-1) < zero {
					result.Set(27, ulpinv)
				}
			}
			if ns1 >= 1 {
				if s1.Get(ns1-1) < zero {
					result.Set(27, ulpinv)
				}
			}

			temp2 = zero
			for j = 1; j <= ns1; j++ {
				temp1 = math.Abs(s1.Get(j-1)-s2.Get(j-1)) / math.Max(math.Sqrt(unfl)*math.Max(s1.Get(0), one), ulp*math.Max(math.Abs(s1.Get(0)), math.Abs(s2.Get(0))))
				temp2 = math.Max(temp1, temp2)
			}
			result.Set(33, temp2)

			//           End of Loop -- Check for RESULT(j) > THRESH
		label270:
			;

			for j = 1; j <= 34; j++ {
				if result.Get(j-1) >= thresh {
					if nfail == 0 {
						dlahd2(path)
					}
					t.Fail()
					fmt.Printf(" m=%5d, n=%5d, type %2d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, ioldsd, j, result.Get(j-1))
					nfail++
				}
			}
			if !bidiag {
				ntest = ntest + 34
			} else {
				ntest = ntest + 30
			}

		label290:
		}
	}

	//     Summary
	// alasum(path, nfail, ntest, 0)

	return
}
