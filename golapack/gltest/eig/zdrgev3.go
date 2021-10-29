package eig

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zdrgev3 checks the nonsymmetric generalized eigenvalue problem driver
// routine ZGGEV3.
//
// ZGGEV3 computes for a pair of n-by-n nonsymmetric matrices (A,B) the
// generalized eigenvalues and, optionally, the left and right
// eigenvectors.
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
// or a ratio  alpha/beta = w, such that A - w*B is singular.  It is
// usually represented as the pair (alpha,beta), as there is reasonable
// interpretation for beta=0, and even for both being zero.
//
// A right generalized eigenvector corresponding to a generalized
// eigenvalue  w  for a pair of matrices (A,B) is a vector r  such that
// (A - wB) * r = 0.  A left generalized eigenvector is a vector l such
// that l**H * (A - wB) = 0, where l**H is the conjugate-transpose of l.
//
// When zdrgev3 is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each _type of matrix, a pair of matrices (A, B) will be generated
// and used for testing.  For each matrix pair, the following tests
// will be performed and compared with the threshold THRESH.
//
// Results from ZGGEV3:
//
// (1)  max over all left eigenvalue/-vector pairs (alpha/beta,l) of
//
//      | VL**H * (beta A - alpha B) |/( ulp max(|beta A|, |alpha B|) )
//
//      where VL**H is the conjugate-transpose of VL.
//
// (2)  | |VL(i)| - 1 | / ulp and whether largest component real
//
//      VL(i) denotes the i-th column of VL.
//
// (3)  max over all left eigenvalue/-vector pairs (alpha/beta,r) of
//
//      | (beta A - alpha B) * VR | / ( ulp max(|beta A|, |alpha B|) )
//
// (4)  | |VR(i)| - 1 | / ulp and whether largest component real
//
//      VR(i) denotes the i-th column of VR.
//
// (5)  W(full) = W(partial)
//      W(full) denotes the eigenvalues computed when both l and r
//      are also computed, and W(partial) denotes the eigenvalues
//      computed when only W, only W and r, or only W and l are
//      computed.
//
// (6)  VL(full) = VL(partial)
//      VL(full) denotes the left eigenvectors computed when both l
//      and r are computed, and VL(partial) denotes the result
//      when only l is computed.
//
// (7)  VR(full) = VR(partial)
//      VR(full) denotes the right eigenvectors computed when both l
//      and r are also computed, and VR(partial) denotes the result
//      when only l is computed.
//
//
// Test Matrices
// ---- --------
//
// The sizes of the test matrices are specified by an array
// NN(1:NSIZES); the value of each element NN(j) specifies one size.
// The "types" are specified by a logical array DOTYPE( 1:NTYPES ); if
// DOTYPE(j) is .TRUE., then matrix _type "j" will be generated.
// Currently, the list of possible types is:
//
// (1)  ( 0, 0 )         (a pair of zero matrices)
//
// (2)  ( I, 0 )         (an identity and a zero matrix)
//
// (3)  ( 0, I )         (an identity and a zero matrix)
//
// (4)  ( I, I )         (a pair of identity matrices)
//
//         t   t
// (5)  ( J , J  )       (a pair of transposed Jordan blocks)
//
//                                     t                ( I   0  )
// (6)  ( X, Y )         where  X = ( J   0  )  and Y = (      t )
//                                  ( 0   I  )          ( 0   J  )
//                       and I is a k x k identity and J a (k+1)x(k+1)
//                       Jordan block; k=(N-1)/2
//
// (7)  ( D, I )         where D is diag( 0, 1,..., N-1 ) (a diagonal
//                       matrix with those diagonal entries.)
// (8)  ( I, D )
//
// (9)  ( big*D, small*I ) where "big" is near overflow and small=1/big
//
// (10) ( small*D, big*I )
//
// (11) ( big*I, small*D )
//
// (12) ( small*I, big*D )
//
// (13) ( big*D, big*I )
//
// (14) ( small*D, small*I )
//
// (15) ( D1, D2 )        where D1 is diag( 0, 0, 1, ..., N-3, 0 ) and
//                        D2 is diag( 0, N-3, N-4,..., 1, 0, 0 )
//           t   t
// (16) Q ( J , J ) Z     where Q and Z are random orthogonal matrices.
//
// (17) Q ( T1, T2 ) Z    where T1 and T2 are upper triangular matrices
//                        with random O(1) entries above the diagonal
//                        and diagonal entries diag(T1) =
//                        ( 0, 0, 1, ..., N-3, 0 ) and diag(T2) =
//                        ( 0, N-3, N-4,..., 1, 0, 0 )
//
// (18) Q ( T1, T2 ) Z    diag(T1) = ( 0, 0, 1, 1, s, ..., s, 0 )
//                        diag(T2) = ( 0, 1, 0, 1,..., 1, 0 )
//                        s = machine precision.
//
// (19) Q ( T1, T2 ) Z    diag(T1)=( 0,0,1,1, 1-d, ..., 1-(N-5)*d=s, 0 )
//                        diag(T2) = ( 0, 1, 0, 1, ..., 1, 0 )
//
//                                                        N-5
// (20) Q ( T1, T2 ) Z    diag(T1)=( 0, 0, 1, 1, a, ..., a   =s, 0 )
//                        diag(T2) = ( 0, 1, 0, 1, ..., 1, 0, 0 )
//
// (21) Q ( T1, T2 ) Z    diag(T1)=( 0, 0, 1, r1, r2, ..., r(N-4), 0 )
//                        diag(T2) = ( 0, 1, 0, 1, ..., 1, 0, 0 )
//                        where r1,..., r(N-4) are random.
//
// (22) Q ( big*T1, small*T2 ) Z    diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (23) Q ( small*T1, big*T2 ) Z    diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (24) Q ( small*T1, small*T2 ) Z  diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (25) Q ( big*T1, big*T2 ) Z      diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (26) Q ( T1, T2 ) Z     where T1 and T2 are random upper-triangular
//                         matrices.
func zdrgev3(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a, b, s, t, q, z, qe *mat.CMatrix, alpha, beta, alpha1, beta1, work *mat.CVector, lwork int, rwork, result *mat.Vector) (err error) {
	var badnn bool
	var cone, ctemp, czero complex128
	var one, safmax, safmin, ulp, ulpinv, zero float64
	var i, iadd, ierr, in, j, jc, jr, jsize, jtype, maxtyp, maxwrk, minwrk, mtypes, n, n1, nb, nerrs, nmats, nmax, ntestt int
	lasign := []bool{false, false, false, false, false, false, true, false, true, true, false, false, true, true, true, false, true, false, false, false, true, true, true, true, true, false}
	lbsign := []bool{false, false, false, false, false, false, false, true, false, false, true, true, false, false, true, false, true, false, false, false, false, false, false, false, false, false}
	rmagn := vf(4)
	ioldsd := make([]int, 4)
	kadd := []int{0, 0, 0, 0, 3, 2}
	kamagn := []int{1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1}
	katype := []int{0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4, 4, 4, 2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0}
	kazero := []int{1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3, 1, 3, 5, 5, 5, 5, 3, 3, 3, 3, 1}
	kbmagn := []int{1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1}
	kbtype := []int{0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4, 1, 1, -4, 2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0}
	kbzero := []int{1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4, 1, 4, 6, 6, 6, 6, 4, 4, 4, 4, 1}
	kclass := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3}
	ktrian := []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	kz1 := []int{0, 1, 2, 1, 3, 3}
	kz2 := []int{0, 0, 1, 2, 1, 1}

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 26

	badnn = false
	nmax = 1
	for j = 1; j <= nsizes; j++ {
		nmax = max(nmax, nn[j-1])
		if nn[j-1] < 0 {
			badnn = true
		}
	}

	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsizes=%v", nsizes)
	} else if badnn {
		err = fmt.Errorf("badnn: nn=%v", nn)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if thresh < zero {
		err = fmt.Errorf("thresh < zero: thresh=%v", thresh)
	} else if a.Rows <= 1 || a.Rows < nmax {
		err = fmt.Errorf("a.Rows <= 1 || a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if q.Rows <= 1 || q.Rows < nmax {
		err = fmt.Errorf("q.Rows <= 1 || q.Rows < nmax: q.Rows=%v, nmax=%v", q.Rows, nmax)
	} else if qe.Rows <= 1 || qe.Rows < nmax {
		err = fmt.Errorf("qe.Rows <= 1 || qe.Rows < nmax: qe.Rows=%v, nmax=%v", qe.Rows, nmax)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	minwrk = 1
	if err == nil && lwork >= 1 {
		minwrk = nmax * (nmax + 1)
		nb = max(1, ilaenv(1, "Zgeqrf", []byte{' '}, nmax, nmax, -1, -1), ilaenv(1, "Zunmqr", []byte("LC"), nmax, nmax, nmax, -1), ilaenv(1, "Zungqr", []byte{' '}, nmax, nmax, nmax, -1))
		maxwrk = max(2*nmax, nmax*(nb+1), nmax*(nmax+1))
		work.SetRe(0, float64(maxwrk))
	}

	if lwork < minwrk {
		err = fmt.Errorf("lwork < minwrk: lwork=%v, minwrk=%v", lwork, minwrk)
	}

	if err != nil {
		gltest.Xerbla2("zdrgev3", err)
		return
	}

	//     Quick return if possible
	if nsizes == 0 || ntypes == 0 {
		return
	}

	ulp = golapack.Dlamch(Precision)
	safmin = golapack.Dlamch(SafeMinimum)
	safmin = safmin / ulp
	safmax = one / safmin
	safmin, safmax = golapack.Dlabad(safmin, safmax)
	ulpinv = one / ulp

	//     The values RMAGN(2:3) depend on N, see below.
	rmagn.Set(0, zero)
	rmagn.Set(1, one)

	//     Loop over sizes, types
	ntestt = 0
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		n1 = max(1, n)
		rmagn.Set(2, safmax*ulp/float64(n1))
		rmagn.Set(3, safmin*ulpinv*float64(n1))

		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label210
			}
			nmats = nmats + 1

			//           Save ISEED in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Generate test matrices A and B
			//
			//           Description of control parameters:
			//
			//           KZLASS: =1 means w/o rotation, =2 means w/ rotation,
			//                   =3 means random.
			//           KATYPE: the "_type" to be passed to ZLATM4 for computing A.
			//           KAZERO: the pattern of zeros on the diagonal for A:
			//                   =1: ( xxx ), =2: (0, xxx ) =3: ( 0, 0, xxx, 0 ),
			//                   =4: ( 0, xxx, 0, 0 ), =5: ( 0, 0, 1, xxx, 0 ),
			//                   =6: ( 0, 1, 0, xxx, 0 ).  (xxx means a string of
			//                   non-zero entries.)
			//           KAMAGN: the magnitude of the matrix: =0: zero, =1: O(1),
			//                   =2: large, =3: small.
			//           LASIGN: .TRUE. if the diagonal elements of A are to be
			//                   multiplied by a random magnitude 1 number.
			//           KBTYPE, KBZERO, KBMAGN, LBSIGN: the same, but for B.
			//           KTRIAN: =0: don't fill in the upper triangle, =1: do.
			//           KZ1, KZ2, KADD: used to implement KAZERO and KBZERO.
			//           RMAGN: used to implement KAMAGN and KBMAGN.
			if mtypes > maxtyp {
				goto label100
			}
			ierr = 0
			if kclass[jtype-1] < 3 {
				//              Generate A (w/o rotation)
				if abs(katype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Zlaset(Full, n, n, czero, czero, a)
					}
				} else {
					in = n
				}
				zlatm4(katype[jtype-1], in, kz1[kazero[jtype-1]-1], kz2[kazero[jtype-1]-1], lasign[jtype-1], rmagn.Get(kamagn[jtype-1]-0), ulp, rmagn.Get(ktrian[jtype-1]*kamagn[jtype-1]-0), 2, &iseed, a)
				iadd = kadd[kazero[jtype-1]-1]
				if iadd > 0 && iadd <= n {
					a.SetRe(iadd-1, iadd-1, rmagn.Get(kamagn[jtype-1]-0))
				}

				//              Generate B (w/o rotation)
				if abs(kbtype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Zlaset(Full, n, n, czero, czero, b)
					}
				} else {
					in = n
				}
				zlatm4(kbtype[jtype-1], in, kz1[kbzero[jtype-1]-1], kz2[kbzero[jtype-1]-1], lbsign[jtype-1], rmagn.Get(kbmagn[jtype-1]-0), one, rmagn.Get(ktrian[jtype-1]*kbmagn[jtype-1]-0), 2, &iseed, b)
				iadd = kadd[kbzero[jtype-1]-1]
				if iadd != 0 && iadd <= n {
					b.SetRe(iadd-1, iadd-1, rmagn.Get(kbmagn[jtype-1]-0))
				}

				if kclass[jtype-1] == 2 && n > 0 {
					//                 Include rotations
					//
					//                 Generate Q, Z as Householder transformations times
					//                 a diagonal matrix.
					for jc = 1; jc <= n-1; jc++ {
						for jr = jc; jr <= n; jr++ {
							q.Set(jr-1, jc-1, matgen.Zlarnd(3, iseed))
							z.Set(jr-1, jc-1, matgen.Zlarnd(3, iseed))
						}
						*q.GetPtr(jc-1, jc-1), *work.GetPtr(jc - 1) = golapack.Zlarfg(n+1-jc, q.Get(jc-1, jc-1), q.CVector(jc, jc-1, 1))
						work.SetRe(2*n+jc-1, math.Copysign(one, q.GetRe(jc-1, jc-1)))
						q.Set(jc-1, jc-1, cone)
						*z.GetPtr(jc-1, jc-1), *work.GetPtr(n + jc - 1) = golapack.Zlarfg(n+1-jc, z.Get(jc-1, jc-1), z.CVector(jc, jc-1, 1))
						work.SetRe(3*n+jc-1, math.Copysign(one, z.GetRe(jc-1, jc-1)))
						z.Set(jc-1, jc-1, cone)
					}
					ctemp = matgen.Zlarnd(3, iseed)
					q.Set(n-1, n-1, cone)
					work.Set(n-1, czero)
					work.Set(3*n-1, ctemp/toCmplx(cmplx.Abs(ctemp)))
					ctemp = matgen.Zlarnd(3, iseed)
					z.Set(n-1, n-1, cone)
					work.Set(2*n-1, czero)
					work.Set(4*n-1, ctemp/toCmplx(cmplx.Abs(ctemp)))

					//                 Apply the diagonal matrices
					for jc = 1; jc <= n; jc++ {
						for jr = 1; jr <= n; jr++ {
							a.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.GetConj(3*n+jc-1)*a.Get(jr-1, jc-1))
							b.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.GetConj(3*n+jc-1)*b.Get(jr-1, jc-1))
						}
					}
					if err = golapack.Zunm2r(Left, NoTrans, n, n, n-1, q, work, a, work.Off(2*n)); err != nil {
						goto label90
					}
					if err = golapack.Zunm2r(Right, ConjTrans, n, n, n-1, z, work.Off(n), a, work.Off(2*n)); err != nil {
						goto label90
					}
					if err = golapack.Zunm2r(Left, NoTrans, n, n, n-1, q, work, b, work.Off(2*n)); err != nil {
						goto label90
					}
					if err = golapack.Zunm2r(Right, ConjTrans, n, n, n-1, z, work.Off(n), b, work.Off(2*n)); err != nil {
						goto label90
					}
				}
			} else {
				//              Random matrices
				for jc = 1; jc <= n; jc++ {
					for jr = 1; jr <= n; jr++ {
						a.Set(jr-1, jc-1, rmagn.GetCmplx(kamagn[jtype-1]-0)*matgen.Zlarnd(4, iseed))
						b.Set(jr-1, jc-1, rmagn.GetCmplx(kbmagn[jtype-1]-0)*matgen.Zlarnd(4, iseed))
					}
				}
			}

		label90:
			;

			if ierr != 0 {
				fmt.Printf(" zdrgev3: %s returned info=%6d.\n   n=%6d, jtype=%6d, iseed=%5d\n", "Generator", ierr, n, jtype, ioldsd)
				err = fmt.Errorf("ierr=%v", abs(ierr))
				return
			}

		label100:
			;

			for i = 1; i <= 7; i++ {
				result.Set(i-1, -one)
			}

			//           Call ZGGEV3 to compute eigenvalues and eigenvectors.
			golapack.Zlacpy(Full, n, n, a, s)
			golapack.Zlacpy(Full, n, n, b, t)
			if ierr, err = golapack.Zggev3('V', 'V', n, s, t, alpha, beta, q, z, work, lwork, rwork); err != nil || (ierr != 0 && ierr != n+1) {
				result.Set(0, ulpinv)
				fmt.Printf(" zdrgev3: %s returned info=%6d.\n   n=%6d, jtype=%6d, iseed=%5d\n", "Zggev31", ierr, n, jtype, ioldsd)
				goto label190
			}

			//           Do the tests (1) and (2)
			zget52(true, n, a, b, q, alpha, beta, work, rwork, result)
			if result.Get(1) > thresh {
				fmt.Printf(" zdrgev3: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,   n=%4d, jtype=%3d, iseed=%5d\n", "Left", "Zggev31", result.Get(1), n, jtype, ioldsd)
				err = fmt.Errorf(" zdrgev3: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,   n=%4d, jtype=%3d, iseed=%5d\n", "Left", "Zggev31", result.Get(1), n, jtype, ioldsd)
			}

			//           Do the tests (3) and (4)
			zget52(false, n, a, b, z, alpha, beta, work, rwork, result.Off(2))
			if result.Get(3) > thresh {
				fmt.Printf(" zdrgev3: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,   n=%4d, jtype=%3d, iseed=%5d\n", "Right", "Zggev31", result.Get(3), n, jtype, ioldsd)
				err = fmt.Errorf(" zdrgev3: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,   n=%4d, jtype=%3d, iseed=%5d\n", "Right", "Zggev31", result.Get(3), n, jtype, ioldsd)
			}

			//           Do test (5)
			golapack.Zlacpy(Full, n, n, a, s)
			golapack.Zlacpy(Full, n, n, b, t)
			if ierr, err = golapack.Zggev3('N', 'N', n, s, t, alpha1, beta1, q, z, work, lwork, rwork); err != nil || (ierr != 0 && ierr != n+1) {
				result.Set(0, ulpinv)
				fmt.Printf(" zdrgev3: %s returned info=%6d.\n   n=%6d, jtype=%6d, iseed=%5d\n", "Zggev32", ierr, n, jtype, ioldsd)
				goto label190
			}

			for j = 1; j <= n; j++ {
				if alpha.Get(j-1) != alpha1.Get(j-1) || beta.Get(j-1) != beta1.Get(j-1) {
					result.Set(4, ulpinv)
				}
			}

			//           Do test (6): Compute eigenvalues and left eigenvectors,
			//           and test them
			golapack.Zlacpy(Full, n, n, a, s)
			golapack.Zlacpy(Full, n, n, b, t)
			if ierr, err = golapack.Zggev3('V', 'N', n, s, t, alpha1, beta1, qe, z, work, lwork, rwork); err != nil || (ierr != 0 && ierr != n+1) {
				result.Set(0, ulpinv)
				fmt.Printf(" zdrgev3: %s returned info=%6d.\n   n=%6d, jtype=%6d, iseed=%5d\n", "Zggev33", ierr, n, jtype, ioldsd)
				goto label190
			}

			for j = 1; j <= n; j++ {
				if alpha.Get(j-1) != alpha1.Get(j-1) || beta.Get(j-1) != beta1.Get(j-1) {
					result.Set(5, ulpinv)
				}
			}

			for j = 1; j <= n; j++ {
				for jc = 1; jc <= n; jc++ {
					if q.Get(j-1, jc-1) != qe.Get(j-1, jc-1) {
						result.Set(5, ulpinv)
					}
				}
			}

			//           Do test (7): Compute eigenvalues and right eigenvectors,
			//           and test them
			golapack.Zlacpy(Full, n, n, a, s)
			golapack.Zlacpy(Full, n, n, b, t)
			if ierr, err = golapack.Zggev3('N', 'V', n, s, t, alpha1, beta1, q, qe, work, lwork, rwork); err != nil || (ierr != 0 && ierr != n+1) {
				result.Set(0, ulpinv)
				fmt.Printf(" zdrgev3: %s returned info=%6d.\n   n=%6d, jtype=%6d, iseed=%5d\n", "Zggev34", ierr, n, jtype, ioldsd)
				goto label190
			}

			for j = 1; j <= n; j++ {
				if alpha.Get(j-1) != alpha1.Get(j-1) || beta.Get(j-1) != beta1.Get(j-1) {
					result.Set(6, ulpinv)
				}
			}

			for j = 1; j <= n; j++ {
				for jc = 1; jc <= n; jc++ {
					if z.Get(j-1, jc-1) != qe.Get(j-1, jc-1) {
						result.Set(6, ulpinv)
					}
				}
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label190:
			;

			ntestt = ntestt + 7

			//           Print out tests which fail.
			for jr = 1; jr <= 7; jr++ {
				if result.Get(jr-1) >= thresh {
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Complex Generalized eigenvalue problem driver\n", "ZGV")

						//                    Matrix types
						fmt.Printf(" Matrix types (see zdrgev3 for details): \n")
						fmt.Printf(" Special Matrices:                       (J'=transposed Jordan block)\n   1=(0,0)  2=(I,0)  3=(0,I)  4=(I,I)  5=(J',J')  6=(diag(J',I), diag(I,J'))\n Diagonal Matrices:  ( D=diag(0,1,2,...) )\n   7=(D,I)   9=(large*D, small*I)  11=(large*I, small*D)  13=(large*D, large*I)\n   8=(I,D)  10=(small*D, large*I)  12=(small*I, large*D)  14=(small*D, small*I)\n  15=(D, reversed D)\n")
						fmt.Printf(" Matrices Rotated by Random %s Matrices U, V:\n  16=Transposed Jordan Blocks             19=geometric alpha, beta=0,1\n  17=arithm. alpha&beta                   20=arithmetic alpha, beta=0,1\n  18=clustered alpha, beta=0,1            21=random alpha, beta=0,1\n Large & Small Matrices:\n  22=(large, small)   23=(small,large)    24=(small,small)    25=(large,large)\n  26=random O(1) matrices.\n", "Orthogonal")

						//                    Tests performed
						fmt.Printf("\n Tests performed:    \n 1 = max | ( b A - a B )'*l | / const.,\n 2 = | |VR(i)| - 1 | / ulp,\n 3 = max | ( b A - a B )*r | / const.\n 4 = | |VL(i)| - 1 | / ulp,\n 5 = 0 if W same no matter if r or l computed,\n 6 = 0 if l same no matter if l computed,\n 7 = 0 if r same no matter if r computed,\n \n")

					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is %8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
						err = fmt.Errorf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is %8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is %10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
						err = fmt.Errorf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is %10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}

		label210:
		}
	}

	//     Summary
	alasvm("Zgv3", nerrs, ntestt, 0)

	work.SetRe(0, float64(maxwrk))

	return
}
