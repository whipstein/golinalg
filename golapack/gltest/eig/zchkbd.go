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

// Zchkbd checks the singular value decomposition (SVD) routines.
//
// ZGEBRD reduces a complex general m by n matrix A to real upper or
// lower bidiagonal form by an orthogonal transformation: Q' * A * P = B
// (or A = Q * B * P').  The matrix B is upper bidiagonal if m >= n
// and lower bidiagonal if m < n.
//
// ZUNGBR generates the orthogonal matrices Q and P' from ZGEBRD.
// Note that Q and P are not necessarily square.
//
// ZBDSQR computes the singular value decomposition of the bidiagonal
// matrix B as B = U S V'.  It is called three times to compute
//    1)  B = U S1 V', where S1 is the diagonal matrix of singular
//        values and the columns of the matrices U and V are the left
//        and right singular vectors, respectively, of B.
//    2)  Same as 1), but the singular values are stored in S2 and the
//        singular vectors are not computed.
//    3)  A = (UQ) S (P'V'), the SVD of the original matrix A.
// In addition, ZBDSQR has an option to apply the left orthogonal matrix
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
// Test ZGEBRD and ZUNGBR
//
// (1)   | A - Q B PT | / ( |A| max(M,N) ulp ), PT = P'
//
// (2)   | I - Q' Q | / ( M ulp )
//
// (3)   | I - PT PT' | / ( N ulp )
//
// Test ZBDSQR on bidiagonal matrix B
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
// Test ZBDSQR on matrix A
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
//      (a) ZGEBRD is not called to reduce it to bidiagonal form.
//      (b) the bidiagonal is  min(M,N) x min(M,N); if M<N, the
//          matrix will be lower bidiagonal, otherwise upper.
//      (c) only tests 5--8 and 14 are performed.
//
// A subset of the full set of matrix types may be selected through
// the logical array DOTYPE.
func Zchkbd(nsizes *int, mval *[]int, nval *[]int, ntypes *int, dotype *[]bool, nrhs *int, iseed *[]int, thresh *float64, a *mat.CMatrix, lda *int, bd, be, s1, s2 *mat.Vector, x *mat.CMatrix, ldx *int, y, z, q *mat.CMatrix, ldq *int, pt *mat.CMatrix, ldpt *int, u, vt *mat.CMatrix, work *mat.CVector, lwork *int, rwork *mat.Vector, nout, info *int, t *testing.T) {
	var badmm, badnn, bidiag bool
	var uplo byte
	var cone, czero complex128
	var amninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, two, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, itype, j, jcol, jsize, jtype, log2ui, m, maxtyp, minwrk, mmax, mnmax, mnmin, mq, mtypes, n, nfail, nmax, ntest int
	var err error
	_ = err
	dumma := vf(1)
	result := vf(14)
	ioldsd := make([]int, 4)
	iwork := make([]int, 1)
	kmagn := make([]int, 16)
	kmode := make([]int, 16)
	ktype := make([]int, 16)

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 16
	infot := &gltest.Common.Infoc.Infot

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15] = 1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9, 10
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 0
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 0

	//     Check for errors
	(*info) = 0

	badmm = false
	badnn = false
	mmax = 1
	nmax = 1
	mnmax = 1
	minwrk = 1
	for j = 1; j <= (*nsizes); j++ {
		mmax = max(mmax, (*mval)[j-1])
		if (*mval)[j-1] < 0 {
			badmm = true
		}
		nmax = max(nmax, (*nval)[j-1])
		if (*nval)[j-1] < 0 {
			badnn = true
		}
		mnmax = max(mnmax, min((*mval)[j-1], (*nval)[j-1]))
		minwrk = max(minwrk, 3*((*mval)[j-1]+(*nval)[j-1]), (*mval)[j-1]*((*mval)[j-1]+max((*mval)[j-1], (*nval)[j-1], *nrhs)+1)+(*nval)[j-1]*min((*nval)[j-1], (*mval)[j-1]))
	}

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badmm {
		(*info) = -2
	} else if badnn {
		(*info) = -3
	} else if (*ntypes) < 0 {
		(*info) = -4
	} else if (*nrhs) < 0 {
		(*info) = -6
	} else if (*lda) < mmax {
		(*info) = -11
	} else if (*ldx) < mmax {
		(*info) = -17
	} else if (*ldq) < mmax {
		(*info) = -21
	} else if (*ldpt) < mnmax {
		(*info) = -23
	} else if minwrk > (*lwork) {
		(*info) = -27
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZCHKBD"), -(*info))
		return
	}

	//     Initialize constants
	path := []byte("ZBD")
	nfail = 0
	ntest = 0
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)
	golapack.Dlabad(&unfl, &ovfl)
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	log2ui = int(math.Log(ulpinv) / math.Log(two))
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)
	(*infot) = 0

	//     Loop over sizes, types
	for jsize = 1; jsize <= (*nsizes); jsize++ {
		m = (*mval)[jsize-1]
		n = (*nval)[jsize-1]
		mnmin = min(m, n)
		amninv = one / float64(max(m, n, 1))

		if (*nsizes) != 1 {
			mtypes = min(maxtyp, *ntypes)
		} else {
			mtypes = min(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label170
			}

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			for j = 1; j <= 14; j++ {
				result.Set(j-1, -one)
			}

			uplo = ' '

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

			golapack.Zlaset('F', lda, &n, &czero, &czero, a, lda)
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
				matgen.Zlatms(&mnmin, &mnmin, 'S', iseed, 'N', rwork, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work, &iinfo)

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				matgen.Zlatms(&mnmin, &mnmin, 'S', iseed, 'S', rwork, &imode, &cond, &anorm, &m, &n, 'N', a, lda, work, &iinfo)

			} else if itype == 6 {
				//              Nonsymmetric, singular values specified
				matgen.Zlatms(&m, &n, 'S', iseed, 'N', rwork, &imode, &cond, &anorm, &m, &n, 'N', a, lda, work, &iinfo)

			} else if itype == 7 {
				//              Diagonal, random entries
				matgen.Zlatmr(&mnmin, &mnmin, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(mnmin), func() *int { y := 1; return &y }(), &one, work.Off(2*mnmin), func() *int { y := 1; return &y }(), &one, 'N', &iwork, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, &iwork, &iinfo)

			} else if itype == 8 {
				//              Symmetric, random entries
				matgen.Zlatmr(&mnmin, &mnmin, 'S', iseed, 'S', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(mnmin), func() *int { y := 1; return &y }(), &one, work.Off(m+mnmin), func() *int { y := 1; return &y }(), &one, 'N', &iwork, &m, &n, &zero, &anorm, 'N', a, lda, &iwork, &iinfo)

			} else if itype == 9 {
				//              Nonsymmetric, random entries
				matgen.Zlatmr(&m, &n, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(mnmin), func() *int { y := 1; return &y }(), &one, work.Off(m+mnmin), func() *int { y := 1; return &y }(), &one, 'N', &iwork, &m, &n, &zero, &anorm, 'N', a, lda, &iwork, &iinfo)

			} else if itype == 10 {
				//              Bidiagonal, random entries
				temp1 = -two * math.Log(ulp)
				for j = 1; j <= mnmin; j++ {
					bd.Set(j-1, math.Exp(temp1*matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed)))
					if j < mnmin {
						be.Set(j-1, math.Exp(temp1*matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed)))
					}
				}

				iinfo = 0
				bidiag = true
				if m >= n {
					uplo = 'U'
				} else {
					uplo = 'L'
				}
			} else {
				iinfo = 1
			}

			if iinfo == 0 {
				//              Generate Right-Hand Side
				if bidiag {
					matgen.Zlatmr(&mnmin, nrhs, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(mnmin), func() *int { y := 1; return &y }(), &one, work.Off(2*mnmin), func() *int { y := 1; return &y }(), &one, 'N', &iwork, &mnmin, nrhs, &zero, &one, 'N', y, ldx, &iwork, &iinfo)
				} else {
					matgen.Zlatmr(&m, nrhs, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(m), func() *int { y := 1; return &y }(), &one, work.Off(2*m), func() *int { y := 1; return &y }(), &one, 'N', &iwork, &m, nrhs, &zero, &one, 'N', x, ldx, &iwork, &iinfo)
				}
			}

			//           Error Exit
			if iinfo != 0 {
				fmt.Printf(" ZCHKBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, m, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				return
			}

		label100:
			;

			//           Call ZGEBRD and ZUNGBR to compute B, Q, and P, do tests.
			if !bidiag {
				//              Compute transformations to reduce A to bidiagonal form:
				//              B := Q' * A * P.
				golapack.Zlacpy(' ', &m, &n, a, lda, q, ldq)
				golapack.Zgebrd(&m, &n, q, ldq, bd, be, work, work.Off(mnmin), work.Off(2*mnmin), toPtr((*lwork)-2*mnmin), &iinfo)

				//              Check error code from ZGEBRD.
				if iinfo != 0 {
					fmt.Printf(" ZCHKBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEBRD", iinfo, m, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					return
				}

				golapack.Zlacpy(' ', &m, &n, q, ldq, pt, ldpt)
				if m >= n {
					uplo = 'U'
				} else {
					uplo = 'L'
				}

				//              Generate Q
				mq = m
				if (*nrhs) <= 0 {
					mq = mnmin
				}
				golapack.Zungbr('Q', &m, &mq, &n, q, ldq, work, work.Off(2*mnmin), toPtr((*lwork)-2*mnmin), &iinfo)

				//              Check error code from ZUNGBR.
				if iinfo != 0 {
					fmt.Printf(" ZCHKBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUNGBR(Q)", iinfo, m, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					return
				}

				//              Generate P'
				golapack.Zungbr('P', &mnmin, &n, &m, pt, ldpt, work.Off(mnmin), work.Off(2*mnmin), toPtr((*lwork)-2*mnmin), &iinfo)

				//              Check error code from ZUNGBR.
				if iinfo != 0 {
					fmt.Printf(" ZCHKBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUNGBR(P)", iinfo, m, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					return
				}

				//              Apply Q' to an M by NRHS matrix X:  Y := Q' * X.
				err = goblas.Zgemm(ConjTrans, NoTrans, m, *nrhs, m, cone, q, x, czero, y)

				//              Test 1:  Check the decomposition A := Q * B * PT
				//                   2:  Check the orthogonality of Q
				//                   3:  Check the orthogonality of PT
				Zbdt01(&m, &n, func() *int { y := 1; return &y }(), a, lda, q, ldq, bd, be, pt, ldpt, work, rwork, result.GetPtr(0))
				Zunt01('C', &m, &mq, q, ldq, work, lwork, rwork, result.GetPtr(1))
				Zunt01('R', &mnmin, &n, pt, ldpt, work, lwork, rwork, result.GetPtr(2))
			}

			//           Use ZBDSQR to form the SVD of the bidiagonal matrix B:
			//           B := U * S1 * VT, and compute Z = U' * Y.
			goblas.Dcopy(mnmin, bd.Off(0, 1), s1.Off(0, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), rwork.Off(0, 1))
			}
			golapack.Zlacpy(' ', &m, nrhs, y, ldx, z, ldx)
			golapack.Zlaset('F', &mnmin, &mnmin, &czero, &cone, u, ldpt)
			golapack.Zlaset('F', &mnmin, &mnmin, &czero, &cone, vt, ldpt)

			golapack.Zbdsqr(uplo, &mnmin, &mnmin, &mnmin, nrhs, s1, rwork, vt, ldpt, u, ldpt, z, ldx, rwork.Off(mnmin), &iinfo)

			//           Check error code from ZBDSQR.
			if iinfo != 0 {
				fmt.Printf(" ZCHKBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZBDSQR(vects)", iinfo, m, n, jtype, ioldsd)
				(*info) = abs(iinfo)
				if iinfo < 0 {
					return
				} else {
					result.Set(3, ulpinv)
					goto label150
				}
			}

			//           Use ZBDSQR to compute only the singular values of the
			//           bidiagonal matrix B;  U, VT, and Z should not be modified.
			goblas.Dcopy(mnmin, bd.Off(0, 1), s2.Off(0, 1))
			if mnmin > 0 {
				goblas.Dcopy(mnmin-1, be.Off(0, 1), rwork.Off(0, 1))
			}

			golapack.Zbdsqr(uplo, &mnmin, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), s2, rwork, vt, ldpt, u, ldpt, z, ldx, rwork.Off(mnmin), &iinfo)

			//           Check error code from ZBDSQR.
			if iinfo != 0 {
				fmt.Printf(" ZCHKBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZBDSQR(values)", iinfo, m, n, jtype, ioldsd)
				(*info) = abs(iinfo)
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
			Zbdt03(uplo, &mnmin, func() *int { y := 1; return &y }(), bd, be, u, ldpt, s1, vt, ldpt, work, result.GetPtr(3))
			Zbdt02(&mnmin, nrhs, y, ldx, z, ldx, u, ldpt, work, rwork, result.GetPtr(4))
			Zunt01('C', &mnmin, &mnmin, u, ldpt, work, lwork, rwork, result.GetPtr(5))
			Zunt01('R', &mnmin, &mnmin, vt, ldpt, work, lwork, rwork, result.GetPtr(6))

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

			//           Test 9:  Compare ZBDSQR with and without singular vectors
			temp2 = zero

			for j = 1; j <= mnmin; j++ {
				temp1 = math.Abs(s1.Get(j-1)-s2.Get(j-1)) / math.Max(math.Sqrt(unfl)*math.Max(s1.Get(0), one), ulp*math.Max(s1.GetMag(j-1), s2.GetMag(j-1)))
				temp2 = math.Max(temp1, temp2)
			}

			result.Set(8, temp2)

			//           Test 10:  Sturm sequence test of singular values
			//                     Go up by factors of two until it succeeds
			temp1 = (*thresh) * (half - ulp)

			for j = 0; j <= log2ui; j++ {
				Dsvdch(&mnmin, bd, be, s1, &temp1, &iinfo)
				if iinfo == 0 {
					goto label140
				}
				temp1 = temp1 * two
			}

		label140:
			;
			result.Set(9, temp1)

			//           Use ZBDSQR to form the decomposition A := (QU) S (VT PT)
			//           from the bidiagonal form A := Q B PT.
			if !bidiag {
				goblas.Dcopy(mnmin, bd.Off(0, 1), s2.Off(0, 1))
				if mnmin > 0 {
					goblas.Dcopy(mnmin-1, be.Off(0, 1), rwork.Off(0, 1))
				}

				golapack.Zbdsqr(uplo, &mnmin, &n, &m, nrhs, s2, rwork, pt, ldpt, q, ldq, y, ldx, rwork.Off(mnmin), &iinfo)

				//              Test 11:  Check the decomposition A := Q*U * S2 * VT*PT
				//                   12:  Check the computation Z := U' * Q' * X
				//                   13:  Check the orthogonality of Q*U
				//                   14:  Check the orthogonality of VT*PT
				Zbdt01(&m, &n, func() *int { y := 0; return &y }(), a, lda, q, ldq, s2, dumma, pt, ldpt, work, rwork, result.GetPtr(10))
				Zbdt02(&m, nrhs, x, ldx, y, ldx, q, ldq, work, rwork, result.GetPtr(11))
				Zunt01('C', &m, &mq, q, ldq, work, lwork, rwork, result.GetPtr(12))
				Zunt01('R', &mnmin, &n, pt, ldpt, work, lwork, rwork, result.GetPtr(13))
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label150:
			;
			for j = 1; j <= 14; j++ {
				if result.Get(j-1) >= (*thresh) {
					if nfail == 0 {
						Dlahd2(path)
					}
					fmt.Printf(" M=%5d, N=%5d, _type %2d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, ioldsd, j, result.Get(j-1))
					nfail = nfail + 1
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
	Alasum(path, &nfail, &ntest, func() *int { y := 0; return &y }())
}
