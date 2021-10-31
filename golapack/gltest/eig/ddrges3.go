package eig

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// ddrges3 checks the nonsymmetric generalized eigenvalue (Schur form)
// problem driver Dgges3.
//
// Dgges3 factors A and B as Q S Z'  and Q T Z' , where ' means
// transpose, T is upper triangular, S is in generalized Schur form
// (block upper triangular, with 1x1 and 2x2 blocks on the diagonal,
// the 2x2 blocks corresponding to complex conjugate pairs of
// generalized eigenvalues), and Q and Z are orthogonal. It also
// computes the generalized eigenvalues (alpha(j),beta(j)), j=1,...,n,
// Thus, w(j) = alpha(j)/beta(j) is a root of the characteristic
// equation
//                 det( A - w(j) B ) = 0
// Optionally it also reorder the eigenvalues so that a selected
// cluster of eigenvalues appears in the leading diagonal block of the
// Schur forms.
//
// When ddrges3 is called, a number of matrix "sizes" ("N's") and a
// number of matrix "TYPES" are specified.  For each size ("N")
// and each TYPE of matrix, a pair of matrices (A, B) will be generated
// and used for testing. For each matrix pair, the following 13 tests
// will be performed and compared with the threshold THRESH except
// the tests (5), (11) and (13).
//
//
// (1)   | A - Q S Z' | / ( |A| n ulp ) (no sorting of eigenvalues)
//
//
// (2)   | B - Q T Z' | / ( |B| n ulp ) (no sorting of eigenvalues)
//
//
// (3)   | I - QQ' | / ( n ulp ) (no sorting of eigenvalues)
//
//
// (4)   | I - ZZ' | / ( n ulp ) (no sorting of eigenvalues)
//
// (5)   if A is in Schur form (i.e. quasi-triangular form)
//       (no sorting of eigenvalues)
//
// (6)   if eigenvalues = diagonal blocks of the Schur form (S, T),
//       i.e., test the maximum over j of D(j)  where:
//
//       if alpha(j) is real:
//                     |alpha(j) - S(j,j)|        |beta(j) - T(j,j)|
//           D(j) = ------------------------ + -----------------------
//                  max(|alpha(j)|,|S(j,j)|)   max(|beta(j)|,|T(j,j)|)
//
//       if alpha(j) is complex:
//                                 | det( s S - w T ) |
//           D(j) = ---------------------------------------------------
//                  ulp max( s norm(S), |w| norm(T) )*norm( s S - w T )
//
//       and S and T are here the 2 x 2 diagonal blocks of S and T
//       corresponding to the j-th and j+1-th eigenvalues.
//       (no sorting of eigenvalues)
//
// (7)   | (A,B) - Q (S,T) Z' | / ( | (A,B) | n ulp )
//            (with sorting of eigenvalues).
//
// (8)   | I - QQ' | / ( n ulp ) (with sorting of eigenvalues).
//
// (9)   | I - ZZ' | / ( n ulp ) (with sorting of eigenvalues).
//
// (10)  if A is in Schur form (i.e. quasi-triangular form)
//       (with sorting of eigenvalues).
//
// (11)  if eigenvalues = diagonal blocks of the Schur form (S, T),
//       i.e. test the maximum over j of D(j)  where:
//
//       if alpha(j) is real:
//                     |alpha(j) - S(j,j)|        |beta(j) - T(j,j)|
//           D(j) = ------------------------ + -----------------------
//                  max(|alpha(j)|,|S(j,j)|)   max(|beta(j)|,|T(j,j)|)
//
//       if alpha(j) is complex:
//                                 | det( s S - w T ) |
//           D(j) = ---------------------------------------------------
//                  ulp max( s norm(S), |w| norm(T) )*norm( s S - w T )
//
//       and S and T are here the 2 x 2 diagonal blocks of S and T
//       corresponding to the j-th and j+1-th eigenvalues.
//       (with sorting of eigenvalues).
//
// (12)  if sorting worked and SDIM is the number of eigenvalues
//       which were SELECTed.
//
// Test Matrices
// =============
//
// The sizes of the test matrices are specified by an array
// NN(1:NSIZES); the value of each element NN(j) specifies one size.
// The "types" are specified by a logical array DOTYPE( 1:NTYPES ); if
// DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
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
func ddrges3(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, nounit int, a, b, s, t, q, z *mat.Matrix, alphar, alphai, beta, work *mat.Vector, lwork int, result *mat.Vector, bwork []bool, _t *testing.T) (nerrs, ntestt int, err error) {
	var badnn, ilabad bool
	var sort byte
	var one, safmax, safmin, temp1, temp2, ulp, ulpinv, zero float64
	var i, i1, iadd, ierr, iinfo, in, isort, j, jc, jr, jsize, jtype, knteig, maxtyp, maxwrk, minwrk, mtypes, n, n1, nb, nmats, nmax, ntest, rsub, sdim int

	rmagn := vf(4)
	iasign := []int{0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0}
	ibsign := []int{0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}
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
	maxtyp = 26

	//     Check for errors

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
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	minwrk = 1
	if err == nil && lwork >= 1 {
		minwrk = max(10*(nmax+1), 3*nmax*nmax)
		nb = max(1, ilaenv(1, "Dgeqrf", []byte{' '}, nmax, nmax, -1, -1), ilaenv(1, "Dormqr", []byte("LT"), nmax, nmax, nmax, -1), ilaenv(1, "Dorgqr", []byte{' '}, nmax, nmax, nmax, -1))
		maxwrk = max(10*(nmax+1), 2*nmax+nmax*nb, 3*nmax*nmax)
		work.Set(0, float64(maxwrk))
	}

	if lwork < minwrk {
		err = fmt.Errorf("lwork < minwrk: lwork=%v, minwrk=%v", lwork, minwrk)
	}

	if err != nil {
		gltest.Xerbla2("ddrges3", err)
		return
	}

	//     Quick return if possible
	if nsizes == 0 || ntypes == 0 {
		return
	}

	safmin = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	safmin = safmin / ulp
	safmax = one / safmin
	safmin, safmax = golapack.Dlabad(safmin, safmax)
	ulpinv = one / ulp

	//     The values RMAGN(2:3) depend on N, see below.
	rmagn.Set(0, zero)
	rmagn.Set(1, one)

	//     Loop over matrix sizes
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

		//        Loop over matrix types
		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label180
			}
			nmats = nmats + 1
			ntest = 0

			//           Save iseed in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Initialize RESULT
			for j = 1; j <= 13; j++ {
				result.Set(j-1, zero)
			}

			//           Generate test matrices A and B
			//
			//           Description of control parameters:
			//
			//           KZLASS: =1 means w/o rotation, =2 means w/ rotation,
			//                   =3 means random.
			//           KATYPE: the "type" to be passed to DLATM4 for computing A.
			//           KAZERO: the pattern of zeros on the diagonal for A:
			//                   =1: ( xxx ), =2: (0, xxx ) =3: ( 0, 0, xxx, 0 ),
			//                   =4: ( 0, xxx, 0, 0 ), =5: ( 0, 0, 1, xxx, 0 ),
			//                   =6: ( 0, 1, 0, xxx, 0 ).  (xxx means a string of
			//                   non-zero entries.)
			//           KAMAGN: the magnitude of the matrix: =0: zero, =1: O(1),
			//                   =2: large, =3: small.
			//           IASIGN: 1 if the diagonal elements of A are to be
			//                   multiplied by a random magnitude 1 number, =2 if
			//                   randomly chosen diagonal blocks are to be rotated
			//                   to form 2x2 blocks.
			//           KBTYPE, KBZERO, KBMAGN, IBSIGN: the same, but for B.
			//           KTRIAN: =0: don't fill in the upper triangle, =1: do.
			//           KZ1, KZ2, KADD: used to implement KAZERO and KBZERO.
			//           RMAGN: used to implement KAMAGN and KBMAGN.
			if mtypes > maxtyp {
				goto label110
			}
			iinfo = 0
			if kclass[jtype-1] < 3 {
				//              Generate A (w/o rotation)
				if abs(katype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Dlaset(Full, n, n, zero, zero, a)
					}
				} else {
					in = n
				}
				dlatm4(katype[jtype-1], in, kz1[kazero[jtype-1]-1], kz2[kazero[jtype-1]-1], iasign[jtype-1], rmagn.Get(kamagn[jtype-1]-0), ulp, rmagn.Get(ktrian[jtype-1]*kamagn[jtype-1]-0), 2, &iseed, a)
				iadd = kadd[kazero[jtype-1]-1]
				if iadd > 0 && iadd <= n {
					a.Set(iadd-1, iadd-1, one)
				}

				//              Generate B (w/o rotation)
				if abs(kbtype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Dlaset(Full, n, n, zero, zero, b)
					}
				} else {
					in = n
				}
				dlatm4(kbtype[jtype-1], in, kz1[kbzero[jtype-1]-1], kz2[kbzero[jtype-1]-1], ibsign[jtype-1], rmagn.Get(kbmagn[jtype-1]-0), one, rmagn.Get(ktrian[jtype-1]*kbmagn[jtype-1]-0), 2, &iseed, b)
				iadd = kadd[kbzero[jtype-1]-1]
				if iadd != 0 && iadd <= n {
					b.Set(iadd-1, iadd-1, one)
				}

				if kclass[jtype-1] == 2 && n > 0 {
					//                 Include rotations
					//
					//                 Generate Q, Z as Householder transformations times
					//                 a diagonal matrix.
					for jc = 1; jc <= n-1; jc++ {
						for jr = jc; jr <= n; jr++ {
							q.Set(jr-1, jc-1, matgen.Dlarnd(3, &iseed))
							z.Set(jr-1, jc-1, matgen.Dlarnd(3, &iseed))
						}
						*q.GetPtr(jc-1, jc-1), *work.GetPtr(jc - 1) = golapack.Dlarfg(n+1-jc, q.Get(jc-1, jc-1), q.Vector(jc, jc-1, 1))
						work.Set(2*n+jc-1, math.Copysign(one, q.Get(jc-1, jc-1)))
						q.Set(jc-1, jc-1, one)
						*z.GetPtr(jc-1, jc-1), *work.GetPtr(n + jc - 1) = golapack.Dlarfg(n+1-jc, z.Get(jc-1, jc-1), z.Vector(jc, jc-1, 1))
						work.Set(3*n+jc-1, math.Copysign(one, z.Get(jc-1, jc-1)))
						z.Set(jc-1, jc-1, one)
					}
					q.Set(n-1, n-1, one)
					work.Set(n-1, zero)
					work.Set(3*n-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))
					z.Set(n-1, n-1, one)
					work.Set(2*n-1, zero)
					work.Set(4*n-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))

					//                 Apply the diagonal matrices
					for jc = 1; jc <= n; jc++ {
						for jr = 1; jr <= n; jr++ {
							a.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.Get(3*n+jc-1)*a.Get(jr-1, jc-1))
							b.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.Get(3*n+jc-1)*b.Get(jr-1, jc-1))
						}
					}
					if err = golapack.Dorm2r(Left, NoTrans, n, n, n-1, q, work, a, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Dorm2r(Right, Trans, n, n, n-1, z, work.Off(n), a, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Dorm2r(Left, NoTrans, n, n, n-1, q, work, b, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Dorm2r(Right, Trans, n, n, n-1, z, work.Off(n), b, work.Off(2*n)); err != nil {
						goto label100
					}
				}
			} else {
				//              Random matrices
				for jc = 1; jc <= n; jc++ {
					for jr = 1; jr <= n; jr++ {
						a.Set(jr-1, jc-1, rmagn.Get(kamagn[jtype-1]-0)*matgen.Dlarnd(2, &iseed))
						b.Set(jr-1, jc-1, rmagn.Get(kbmagn[jtype-1]-0)*matgen.Dlarnd(2, &iseed))
					}
				}
			}

		label100:
			;

			if iinfo != 0 {
				fmt.Printf(" ddrges3: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%4d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label110:
			;

			for i = 1; i <= 13; i++ {
				result.Set(i-1, -one)
			}

			//           Test with and without sorting of eigenvalues
			for isort = 0; isort <= 1; isort++ {
				if isort == 0 {
					sort = 'N'
					rsub = 0
				} else {
					sort = 'S'
					rsub = 5
				}

				//              Call Dgges3 to compute H, T, Q, Z, alpha, and beta.
				golapack.Dlacpy(Full, n, n, a, s)
				golapack.Dlacpy(Full, n, n, b, t)
				ntest = 1 + rsub + isort
				result.Set(1+rsub+isort-1, ulpinv)
				if sdim, iinfo, err = golapack.Dgges3('V', 'V', sort, dlctes, n, s, t, alphar, alphai, beta, q, z, work, lwork, &bwork); iinfo != 0 && iinfo != n+2 || err != nil {
					_t.Fail()
					result.Set(1+rsub+isort-1, ulpinv)
					fmt.Printf(" ddrges3: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%4d\n", "Dgges3", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					goto label160
				}

				ntest = 4 + rsub

				//              Do tests 1--4 (or tests 7--9 when reordering )
				if isort == 0 {
					result.Set(0, dget51(1, n, a, s, q, z, work))
					result.Set(1, dget51(1, n, b, t, q, z, work))
				} else {
					result.Set(6, Dget54(n, a, b, s, t, q, z, work))
				}
				result.Set(3+rsub-1, dget51(3, n, a, t, q, q, work))
				result.Set(4+rsub-1, dget51(3, n, b, t, z, z, work))

				//              Do test 5 and 6 (or Tests 10 and 11 when reordering):
				//              check Schur form of A and compare eigenvalues with
				//              diagonals.
				ntest = 6 + rsub
				temp1 = zero

				for j = 1; j <= n; j++ {
					ilabad = false
					if alphai.Get(j-1) == zero {
						temp2 = (math.Abs(alphar.Get(j-1)-s.Get(j-1, j-1))/math.Max(safmin, math.Max(math.Abs(alphar.Get(j-1)), math.Abs(s.Get(j-1, j-1)))) + math.Abs(beta.Get(j-1)-t.Get(j-1, j-1))/math.Max(safmin, math.Max(math.Abs(beta.Get(j-1)), math.Abs(t.Get(j-1, j-1))))) / ulp

						if j < n {
							if s.Get(j, j-1) != zero {
								ilabad = true
								result.Set(5+rsub-1, ulpinv)
							}
						}
						if j > 1 {
							if s.Get(j-1, j-1-1) != zero {
								ilabad = true
								result.Set(5+rsub-1, ulpinv)
							}
						}

					} else {
						if alphai.Get(j-1) > zero {
							i1 = j
						} else {
							i1 = j - 1
						}
						if i1 <= 0 || i1 >= n {
							ilabad = true
						} else if i1 < n-1 {
							if s.Get(i1+2-1, i1) != zero {
								ilabad = true
								result.Set(5+rsub-1, ulpinv)
							}
						} else if i1 > 1 {
							if s.Get(i1-1, i1-1-1) != zero {
								ilabad = true
								result.Set(5+rsub-1, ulpinv)
							}
						}
						if !ilabad {
							temp2, ierr = dget53(s.Off(i1-1, i1-1), t.Off(i1-1, i1-1), beta.Get(j-1), alphar.Get(j-1), alphai.Get(j-1))
							if ierr >= 3 {
								_t.Fail()
								fmt.Printf(" ddrges3: DGET53 returned info=%1d for eigenvalue %6d.\n         n=%6d, jtype=%6d, iseed=%4d\n", ierr, j, n, jtype, ioldsd)
								err = fmt.Errorf("iinfo=%v", abs(iinfo))
							}
						} else {
							temp2 = ulpinv
						}

					}
					temp1 = math.Max(temp1, temp2)
					if ilabad {
						_t.Fail()
						fmt.Printf(" ddrges3: S not in Schur form at eigenvalue %6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", j, n, jtype, ioldsd)
					}
				}
				result.Set(6+rsub-1, temp1)

				if isort >= 1 {
					//                 Do test 12
					ntest = 12
					result.Set(11, zero)
					knteig = 0
					for i = 1; i <= n; i++ {
						if dlctes(alphar.GetPtr(i-1), alphai.GetPtr(i-1), beta.GetPtr(i-1)) || dlctes(alphar.GetPtr(i-1), toPtrf64(-alphai.Get(i-1)), beta.GetPtr(i-1)) {
							knteig = knteig + 1
						}
						if i < n {
							if (dlctes(alphar.GetPtr(i), alphai.GetPtr(i), beta.GetPtr(i)) || dlctes(alphar.GetPtr(i), toPtrf64(-alphai.Get(i)), beta.GetPtr(i))) && (!(dlctes(alphar.GetPtr(i-1), alphai.GetPtr(i-1), beta.GetPtr(i-1)) || dlctes(alphar.GetPtr(i-1), toPtrf64(-alphai.Get(i-1)), beta.GetPtr(i-1)))) && iinfo != n+2 {
								result.Set(11, ulpinv)
							}
						}
					}
					if sdim != knteig {
						result.Set(11, ulpinv)
					}
				}

			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label160:
			;

			ntestt = ntestt + ntest

			//           Print out tests which fail.
			for jr = 1; jr <= ntest; jr++ {
				if result.Get(jr-1) >= thresh {
					_t.Fail()
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Real Generalized Schur form driver\n", "DGS")

						//                    Matrix types
						fmt.Printf(" Matrix types (see ddrges3 for details): \n")
						fmt.Printf(" Special Matrices:                       (J'=transposed Jordan block)\n   1=(0,0)  2=(I,0)  3=(0,I)  4=(I,I)  5=(J',J')  6=(diag(J',I), diag(I,J'))\n Diagonal Matrices:  ( D=diag(0,1,2,...) )\n   7=(D,I)   9=(large*D, small*I)  11=(large*I, small*D)  13=(large*D, large*I)\n   8=(I,D)  10=(small*D, large*I)  12=(small*I, large*D)  14=(small*D, small*I)\n  15=(D, reversed D)\n")
						fmt.Printf(" Matrices Rotated by Random %s Matrices U, V:\n  16=Transposed Jordan Blocks             19=geometric alpha, beta=0,1\n  17=arithm. alpha&beta                   20=arithmetic alpha, beta=0,1\n  18=clustered alpha, beta=0,1            21=random alpha, beta=0,1\n Large & Small Matrices:\n  22=(large, small)   23=(small,large)    24=(small,small)    25=(large,large)\n  26=random O(1) matrices.\n", "Orthogonal")

						//                    Tests performed
						fmt.Printf("\n Tests performed:  (S is Schur, T is triangular, Q and Z are %s,\n                   l and r are the appropriate left and right\n                   eigenvectors, resp., a is alpha, b is beta, and\n                   %s means %s.)\n Without ordering: \n  1 = | A - Q S Z%s | / ( |A| n ulp )      2 = | B - Q T Z%s | / ( |B| n ulp )\n  3 = | I - QQ%s | / ( n ulp )             4 = | I - ZZ%s | / ( n ulp )\n  5 = A is in Schur form S\n  6 = difference between (alpha,beta) and diagonals of (S,T)\n With ordering: \n  7 = | (A,B) - Q (S,T) Z%s | / ( |(A,B)| n ulp )  \n  8 = | I - QQ%s | / ( n ulp )            9 = | I - ZZ%s | / ( n ulp )\n 10 = A is in Schur form S\n 11 = difference between (alpha,beta) and diagonals of (S,T)\n 12 = SDIM is the correct number of selected eigenvalues\n\n", "orthogonal", "'", "transpose", "'", "'", "'", "'", "'", "'", "'")

					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, type=%2d, seed=%4d, result %2d is %8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, type=%2d, seed=%4d, result %2d is %10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}

		label180:
		}
	}

	//     Summary
	// alasvm("Dgs", nerrs, ntestt, 0)

	work.Set(0, float64(maxwrk))

	return
}
