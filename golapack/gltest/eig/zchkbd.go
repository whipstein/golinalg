package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchkbd checks the singular value decomposition (SVD) routines.
//
// Zgebrd reduces a complex general m by n matrix A to real upper or
// lower bidiagonal form by an orthogonal transformation: Q' * A * P = B
// (or A = Q * B * P').  The matrix B is upper bidiagonal if m >= n
// and lower bidiagonal if m < n.
//
// Zungbr generates the orthogonal matrices Q and P' from Zgebrd.
// Note that Q and P are not necessarily square.
//
// Zbdsqr computes the singular value decomposition of the bidiagonal
// matrix B as B = U S V'.  It is called three times to compute
//    1)  B = U S1 V', where S1 is the diagonal matrix of singular
//        values and the columns of the matrices U and V are the left
//        and right singular vectors, respectively, of B.
//    2)  Same as 1), but the singular values are stored in S2 and the
//        singular vectors are not computed.
//    3)  A = (UQ) S (P'V'), the SVD of the original matrix A.
// In addition, Zbdsqr has an option to apply the left orthogonal matrix
// U to a matrix X, useful in least squares applications.
//
// For each pair of matrix dimensions (M,N) and each selected matrix
// _type, an M by N matrix A and an M by NRHS matrix X are generated.
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
// Test Zgebrd and Zungbr
//
// (1)   | A - Q B PT | / ( |A| max(M,N) ulp ), PT = P'
//
// (2)   | I - Q' Q | / ( M ulp )
//
// (3)   | I - PT PT' | / ( N ulp )
//
// Test Zbdsqr on bidiagonal matrix B
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
// (9)   0 if the true singular values of B are within THRESH of
//       those in S1.  2*THRESH if they are not.  (Tested using
//       DSVDCH)
//
// (10)  | S1 - S2 | / ( |S1| ulp ), where S2 is computed without
//                                   computing U and V.
//
// Test Zbdsqr on matrix A
//
// (11)  | A - (QU) S (VT PT) | / ( |A| max(M,N) ulp )
//
// (12)  | X - (QU) Z | / ( |X| max(M,k) ulp )
//
// (13)  | I - (QU)'(QU) | / ( M ulp )
//
// (14)  | I - (VT PT) (PT'VT') | / ( N ulp )
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
// (6)  Same as (3), but multiplied by SQRT( overflow threshold )
// (7)  Same as (3), but multiplied by SQRT( underflow threshold )
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
// (11) Same as (8), but multiplied by SQRT( overflow threshold )
// (12) Same as (8), but multiplied by SQRT( underflow threshold )
//
// (13) Rectangular matrix with random entries chosen from (-1,1).
// (14) Same as (13), but multiplied by SQRT( overflow threshold )
// (15) Same as (13), but multiplied by SQRT( underflow threshold )
//
// Special case:
// (16) A bidiagonal matrix with random entries chosen from a
//      logarithmic distribution on [ulp^2,ulp^(-2)]  (I.e., each
//      entry is  e^x, where x is chosen uniformly on
//      [ 2 log(ulp), -2 log(ulp) ] .)  For *this* _type:
//      (a) Zgebrd is not called to reduce it to bidiagonal form.
//      (b) the bidiagonal is  min(M,N) x min(M,N); if M<N, the
//          matrix will be lower bidiagonal, otherwise upper.
//      (c) only tests 5--8 and 14 are performed.
//
// A subset of the full set of matrix types may be selected through
// the logical array DOTYPE.
func zchkbd(nsizes int, mval, nval []int, ntypes int, dotype []bool, nrhs int, iseed *[]int, thresh float64, a *mat.CMatrix, bd, be, s1, s2 *mat.Vector, x, y, z, q, pt, u, vt *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector) (nfail, ntest int, err error) {
	var badmm, badnn, bidiag bool
	var uplo mat.MatUplo
	var cone, czero complex128
	var amninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, two, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, itype, j, jcol, jsize, jtype, log2ui, m, maxtyp, minwrk, mmax, mnmax, mnmin, mq, mtypes, n, nmax int

	dumma := vf(1)
	result := vf(14)
	ioldsd := make([]int, 4)
	iwork := make([]int, 1)
	kmagn := []int{1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 0}
	kmode := []int{0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 0}
	ktype := []int{1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9, 10}

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 16
	infot := &gltest.Common.Infoc.Infot

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
		err = fmt.Errorf("badnn: nval=%v", nval)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < mmax {
		err = fmt.Errorf("a.Rows < mmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if x.Rows < mmax {
		err = fmt.Errorf("x.Rows < mmax: x.Rows=%v, nmax=%v", x.Rows, nmax)
	} else if q.Rows < mmax {
		err = fmt.Errorf("q.Rows < mmax: q.Rows=%v, nmax=%v", q.Rows, nmax)
	} else if pt.Rows < mnmax {
		err = fmt.Errorf("pt.Rows < mmax: pt.Rows=%v, nmax=%v", pt.Rows, nmax)
	} else if minwrk > lwork {
		err = fmt.Errorf("minwrk > lwork: minwrk=%v, lwork=%v", minwrk, lwork)
	}

	if err != nil {
		gltest.Xerbla2("zchkbd", err)
		return
	}

	//     Initialize constants
	path := "Zbd"
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
				goto label170
			}

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			for j = 1; j <= 14; j++ {
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
			//       =10                     random bidiagonal (log. distrib.)
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

			golapack.Zlaset(Full, a.Rows, n, czero, czero, a)
			iinfo = 0
			cond = ulpinv

			bidiag = false
			if itype == 1 {
				//              Zero matrix
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jcol = 1; jcol <= mnmin; jcol++ {
					a.SetRe(jcol-1, jcol-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				err = matgen.Zlatms(mnmin, mnmin, 'S', iseed, 'N', rwork, imode, cond, anorm, 0, 0, 'N', a, work)

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				err = matgen.Zlatms(mnmin, mnmin, 'S', iseed, 'S', rwork, imode, cond, anorm, m, n, 'N', a, work)

			} else if itype == 6 {
				//              Nonsymmetric, singular values specified
				err = matgen.Zlatms(m, n, 'S', iseed, 'N', rwork, imode, cond, anorm, m, n, 'N', a, work)

			} else if itype == 7 {
				//              Diagonal, random entries
				err = matgen.Zlatmr(mnmin, mnmin, 'S', iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(mnmin), 1, one, work.Off(2*mnmin), 1, one, 'N', &iwork, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Symmetric, random entries
				err = matgen.Zlatmr(mnmin, mnmin, 'S', iseed, 'S', work, 6, one, cone, 'T', 'N', work.Off(mnmin), 1, one, work.Off(m+mnmin), 1, one, 'N', &iwork, m, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              Nonsymmetric, random entries
				err = matgen.Zlatmr(m, n, 'S', iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(mnmin), 1, one, work.Off(m+mnmin), 1, one, 'N', &iwork, m, n, zero, anorm, 'N', a, &iwork)

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

			if iinfo == 0 || err != nil {
				//              Generate Right-Hand Side
				if bidiag {
					err = matgen.Zlatmr(mnmin, nrhs, 'S', iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(mnmin), 1, one, work.Off(2*mnmin), 1, one, 'N', &iwork, mnmin, nrhs, zero, one, 'N', y, &iwork)
				} else {
					err = matgen.Zlatmr(m, nrhs, 'S', iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(m), 1, one, work.Off(2*m), 1, one, 'N', &iwork, m, nrhs, zero, one, 'N', x, &iwork)
				}
			}

			//           Error Exit
			if iinfo != 0 || err != nil {
				fmt.Printf(" zchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label100:
			;

			//           Call Zgebrd and Zungbr to compute B, Q, and P, do tests.
			if !bidiag {
				//              Compute transformations to reduce A to bidiagonal form:
				//              B := Q' * A * P.
				golapack.Zlacpy(Full, m, n, a, q)
				if err = golapack.Zgebrd(m, n, q, bd, be, work, work.Off(mnmin), work.Off(2*mnmin), lwork-2*mnmin); err != nil {
					fmt.Printf(" zchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Zgebrd", iinfo, m, n, jtype, ioldsd)
					return
				}

				golapack.Zlacpy(Full, m, n, q, pt)
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
				if err = golapack.Zungbr('Q', m, mq, n, q, work, work.Off(2*mnmin), lwork-2*mnmin); err != nil {
					fmt.Printf(" zchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Zungbr(Q)", iinfo, m, n, jtype, ioldsd)
					return
				}

				//              Generate P'
				if err = golapack.Zungbr('P', mnmin, n, m, pt, work.Off(mnmin), work.Off(2*mnmin), lwork-2*mnmin); err != nil {
					fmt.Printf(" zchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Zungbr(P)", iinfo, m, n, jtype, ioldsd)
					return
				}

				//              Apply Q' to an M by NRHS matrix X:  Y := Q' * X.
				if err = y.Gemm(ConjTrans, NoTrans, m, nrhs, m, cone, q, x, czero); err != nil {
					panic(err)
				}

				//              Test 1:  Check the decomposition A := Q * B * PT
				//                   2:  Check the orthogonality of Q
				//                   3:  Check the orthogonality of PT
				result.Set(0, zbdt01(m, n, 1, a, q, bd, be, pt, work, rwork))
				result.Set(1, zunt01('C', m, mq, q, work, lwork, rwork))
				result.Set(2, zunt01('R', mnmin, n, pt, work, lwork, rwork))
			}

			//           Use Zbdsqr to form the SVD of the bidiagonal matrix B:
			//           B := U * S1 * VT, and compute Z = U' * Y.
			s1.Copy(mnmin, bd, 1, 1)
			if mnmin > 0 {
				rwork.Copy(mnmin-1, be, 1, 1)
			}
			golapack.Zlacpy(Full, m, nrhs, y, z)
			golapack.Zlaset(Full, mnmin, mnmin, czero, cone, u)
			golapack.Zlaset(Full, mnmin, mnmin, czero, cone, vt)

			if iinfo, err = golapack.Zbdsqr(uplo, mnmin, mnmin, mnmin, nrhs, s1, rwork, vt, u, z, rwork.Off(mnmin)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Zbdsqr(vects)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(3, ulpinv)
					goto label150
				}
			}

			//           Use Zbdsqr to compute only the singular values of the
			//           bidiagonal matrix B;  U, VT, and Z should not be modified.
			s2.Copy(mnmin, bd, 1, 1)
			if mnmin > 0 {
				rwork.Copy(mnmin-1, be, 1, 1)
			}

			if iinfo, err = golapack.Zbdsqr(uplo, mnmin, 0, 0, 0, s2, rwork, vt, u, z, rwork.Off(mnmin)); err != nil || iinfo != 0 {
				fmt.Printf(" zchkbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Zbdsqr(values)", iinfo, m, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(8, ulpinv)
					goto label150
				}
			}

			//           Test 4:  Check the decomposition B := U * S1 * VT
			//                5:  Check the computation Z := U' * Y
			//                6:  Check the orthogonality of U
			//                7:  Check the orthogonality of VT
			result.Set(3, zbdt03(uplo, mnmin, 1, bd, be, u, s1, vt, work))
			result.Set(4, zbdt02(mnmin, nrhs, y, z, u, work, rwork))
			result.Set(5, zunt01('C', mnmin, mnmin, u, work, lwork, rwork))
			result.Set(6, zunt01('R', mnmin, mnmin, vt, work, lwork, rwork))

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

			//           Test 9:  Compare Zbdsqr with and without singular vectors
			temp2 = zero

			for j = 1; j <= mnmin; j++ {
				temp1 = math.Abs(s1.Get(j-1)-s2.Get(j-1)) / math.Max(math.Sqrt(unfl)*math.Max(s1.Get(0), one), ulp*math.Max(s1.GetMag(j-1), s2.GetMag(j-1)))
				temp2 = math.Max(temp1, temp2)
			}

			result.Set(8, temp2)

			//           Test 10:  Sturm sequence test of singular values
			//                     Go up by factors of two until it succeeds
			temp1 = thresh * (half - ulp)

			for j = 0; j <= log2ui; j++ {
				iinfo = dsvdch(mnmin, bd, be, s1, temp1)
				if iinfo == 0 {
					goto label140
				}
				temp1 = temp1 * two
			}

		label140:
			;
			result.Set(9, temp1)

			//           Use Zbdsqr to form the decomposition A := (QU) S (VT PT)
			//           from the bidiagonal form A := Q B PT.
			if !bidiag {
				s2.Copy(mnmin, bd, 1, 1)
				if mnmin > 0 {
					rwork.Copy(mnmin-1, be, 1, 1)
				}

				if iinfo, err = golapack.Zbdsqr(uplo, mnmin, n, m, nrhs, s2, rwork, pt, q, y, rwork.Off(mnmin)); err != nil {
					panic(err)
				}

				//              Test 11:  Check the decomposition A := Q*U * S2 * VT*PT
				//                   12:  Check the computation Z := U' * Q' * X
				//                   13:  Check the orthogonality of Q*U
				//                   14:  Check the orthogonality of VT*PT
				result.Set(10, zbdt01(m, n, 0, a, q, s2, dumma, pt, work, rwork))
				result.Set(11, zbdt02(m, nrhs, x, y, q, work, rwork))
				result.Set(12, zunt01('C', m, mq, q, work, lwork, rwork))
				result.Set(13, zunt01('R', mnmin, n, pt, work, lwork, rwork))
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label150:
			;
			for j = 1; j <= 14; j++ {
				if result.Get(j-1) >= thresh {
					if nfail == 0 {
						dlahd2(path)
					}
					fmt.Printf(" m=%5d, n=%5d, _type %2d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, ioldsd, j, result.Get(j-1))
					nfail++
				}
			}
			if !bidiag {
				ntest = ntest + 14
			} else {
				ntest = ntest + 5
			}

		label170:
		}
	}

	//     Summary
	// alasum(path, nfail, ntest, 0)

	return
}
