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

// dchkgg checks the nonsymmetric generalized eigenvalue problem
// routines.
//                                T          T        T
// Dgghrd factors A and B as U H V  and U T V , where   means
// transpose, H is hessenberg, T is triangular and U and V are
// orthogonal.
//                                 T          T
// Dhgeqz factors H and T as  Q S Z  and Q P Z , where P is upper
// triangular, S is in generalized Schur form (block upper triangular,
// with 1x1 and 2x2 blocks on the diagonal, the 2x2 blocks
// corresponding to complex conjugate pairs of generalized
// eigenvalues), and Q and Z are orthogonal.  It also computes the
// generalized eigenvalues (alpha(1),beta(1)),...,(alpha(n),beta(n)),
// where alpha(j)=S(j,j) and beta(j)=P(j,j) -- thus,
// w(j) = alpha(j)/beta(j) is a root of the generalized eigenvalue
// problem
//
//     det( A - w(j) B ) = 0
//
// and m(j) = beta(j)/alpha(j) is a root of the essentially equivalent
// problem
//
//     det( m(j) A - B ) = 0
//
// Dtgevc computes the matrix L of left eigenvectors and the matrix R
// of right eigenvectors for the matrix pair ( S, P ).  In the
// description below,  l and r are left and right eigenvectors
// corresponding to the generalized eigenvalues (alpha,beta).
//
// When dchkgg is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each type of matrix, one matrix will be generated and used
// to test the nonsymmetric eigenroutines.  For each matrix, 15
// tests will be performed.  The first twelve "test ratios" should be
// small -- O(1).  They will be compared with the threshold THRESH:
//
//                  T
// (1)   | A - U H V  | / ( |A| n ulp )
//
//                  T
// (2)   | B - U T V  | / ( |B| n ulp )
//
//               T
// (3)   | I - UU  | / ( n ulp )
//
//               T
// (4)   | I - VV  | / ( n ulp )
//
//                  T
// (5)   | H - Q S Z  | / ( |H| n ulp )
//
//                  T
// (6)   | T - Q P Z  | / ( |T| n ulp )
//
//               T
// (7)   | I - QQ  | / ( n ulp )
//
//               T
// (8)   | I - ZZ  | / ( n ulp )
//
// (9)   math.Max over all left eigenvalue/-vector pairs (beta/alpha,l) of
//
//    | l**H * (beta S - alpha P) | / ( ulp math.Max( |beta S|, |alpha P| ) )
//
// (10)  math.Max over all left eigenvalue/-vector pairs (beta/alpha,l') of
//                           T
//   | l'**H * (beta H - alpha T) | / ( ulp math.Max( |beta H|, |alpha T| ) )
//
//       where the eigenvectors l' are the result of passing Q to
//       Dtgevc and back transforming (howmny='B').
//
// (11)  math.Max over all right eigenvalue/-vector pairs (beta/alpha,r) of
//
//       | (beta S - alpha T) r | / ( ulp math.Max( |beta S|, |alpha T| ) )
//
// (12)  math.Max over all right eigenvalue/-vector pairs (beta/alpha,r') of
//
//       | (beta H - alpha T) r' | / ( ulp math.Max( |beta H|, |alpha T| ) )
//
//       where the eigenvectors r' are the result of passing Z to
//       Dtgevc and back transforming (howmny='B').
//
// The last three test ratios will usually be small, but there is no
// mathematical requirement that they be so.  They are therefore
// compared with THRESH only if TSTDIF is .TRUE.
//
// (13)  | S(Q,Z computed) - S(Q,Z not computed) | / ( |S| ulp )
//
// (14)  | P(Q,Z computed) - P(Q,Z not computed) | / ( |P| ulp )
//
// (15)  math.Max( |alpha(Q,Z computed) - alpha(Q,Z not computed)|/|S| ,
//            |beta(Q,Z computed) - beta(Q,Z not computed)|/|P| ) / ulp
//
// In addition, the normalization of L and R are checked, and compared
// with the threshold THRSHN.
//
// Test Matrices
// ---- --------
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
// (16) U ( J , J ) V     where U and V are random orthogonal matrices.
//
// (17) U ( T1, T2 ) V    where T1 and T2 are upper triangular matrices
//                        with random O(1) entries above the diagonal
//                        and diagonal entries diag(T1) =
//                        ( 0, 0, 1, ..., N-3, 0 ) and diag(T2) =
//                        ( 0, N-3, N-4,..., 1, 0, 0 )
//
// (18) U ( T1, T2 ) V    diag(T1) = ( 0, 0, 1, 1, s, ..., s, 0 )
//                        diag(T2) = ( 0, 1, 0, 1,..., 1, 0 )
//                        s = machine precision.
//
// (19) U ( T1, T2 ) V    diag(T1)=( 0,0,1,1, 1-d, ..., 1-(N-5)*d=s, 0 )
//                        diag(T2) = ( 0, 1, 0, 1, ..., 1, 0 )
//
//                                                        N-5
// (20) U ( T1, T2 ) V    diag(T1)=( 0, 0, 1, 1, a, ..., a   =s, 0 )
//                        diag(T2) = ( 0, 1, 0, 1, ..., 1, 0, 0 )
//
// (21) U ( T1, T2 ) V    diag(T1)=( 0, 0, 1, r1, r2, ..., r(N-4), 0 )
//                        diag(T2) = ( 0, 1, 0, 1, ..., 1, 0, 0 )
//                        where r1,..., r(N-4) are random.
//
// (22) U ( big*T1, small*T2 ) V    diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (23) U ( small*T1, big*T2 ) V    diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (24) U ( small*T1, small*T2 ) V  diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (25) U ( big*T1, big*T2 ) V      diag(T1) = ( 0, 0, 1, ..., N-3, 0 )
//                                  diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (26) U ( T1, T2 ) V     where T1 and T2 are random upper-triangular
//                         matrices.
func dchkgg(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, tstdif bool, thrshn float64, nounit int, a, b, h, t, s1, s2, p1, p2, u, v, q, z *mat.Matrix, alphr1, alphi1, beta1, alphr3, alphi3, beta3 *mat.Vector, evectl, evectr *mat.Matrix, work *mat.Vector, lwork int, llwork []bool, result *mat.Vector, _t *testing.T) (nerrs, ntestt int, err error) {
	var badnn bool
	var anorm, bnorm, one, safmax, safmin, temp1, temp2, ulp, ulpinv, zero float64
	var i1, iadd, iinfo, in, j, jc, jr, jsize, jtype, lwkopt, maxtyp, mtypes, n, n1, nmats, nmax, ntest int

	dumma := vf(4)
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

	//     Maximum blocksize and shift -- we assume that blocksize and number
	//     of shifts are monotone increasing functions of N.
	lwkopt = max(6*nmax, 2*nmax*nmax, 1)

	//     Check for errors
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
	} else if u.Rows <= 1 || u.Rows < nmax {
		err = fmt.Errorf("u.Rows <= 1 || u.Rows < nmax: u.Rows=%v, nmax=%v", u.Rows, nmax)
	} else if lwkopt > lwork {
		err = fmt.Errorf("lwkopt > lwork: lwkopt=%v, lwork=%v", lwkopt, lwork)
	}

	if err != nil {
		gltest.Xerbla2("dchkgg", err)
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
				goto label230
			}
			nmats = nmats + 1
			ntest = 0

			//           Save iseed in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Initialize RESULT
			for j = 1; j <= 15; j++ {
				result.Set(j-1, zero)
			}

			//           Compute A and B
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
					a.Set(iadd-1, iadd-1, rmagn.Get(kamagn[jtype-1]-0))
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
					b.Set(iadd-1, iadd-1, rmagn.Get(kbmagn[jtype-1]-0))
				}

				if kclass[jtype-1] == 2 && n > 0 {
					//                 Include rotations
					//
					//                 Generate U, V as Householder transformations times
					//                 a diagonal matrix.
					for jc = 1; jc <= n-1; jc++ {
						for jr = jc; jr <= n; jr++ {
							u.Set(jr-1, jc-1, matgen.Dlarnd(3, &iseed))
							v.Set(jr-1, jc-1, matgen.Dlarnd(3, &iseed))
						}
						*u.GetPtr(jc-1, jc-1), *work.GetPtr(jc - 1) = golapack.Dlarfg(n+1-jc, u.Get(jc-1, jc-1), u.Off(jc, jc-1).Vector(), 1)
						work.Set(2*n+jc-1, math.Copysign(one, u.Get(jc-1, jc-1)))
						u.Set(jc-1, jc-1, one)
						*v.GetPtr(jc-1, jc-1), *work.GetPtr(n + jc - 1) = golapack.Dlarfg(n+1-jc, v.Get(jc-1, jc-1), v.Off(jc, jc-1).Vector(), 1)
						work.Set(3*n+jc-1, math.Copysign(one, v.Get(jc-1, jc-1)))
						v.Set(jc-1, jc-1, one)
					}
					u.Set(n-1, n-1, one)
					work.Set(n-1, zero)
					work.Set(3*n-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))
					v.Set(n-1, n-1, one)
					work.Set(2*n-1, zero)
					work.Set(4*n-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))

					//                 Apply the diagonal matrices
					for jc = 1; jc <= n; jc++ {
						for jr = 1; jr <= n; jr++ {
							a.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.Get(3*n+jc-1)*a.Get(jr-1, jc-1))
							b.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.Get(3*n+jc-1)*b.Get(jr-1, jc-1))
						}
					}
					if err = golapack.Dorm2r(Left, NoTrans, n, n, n-1, u, work, a, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Dorm2r(Right, Trans, n, n, n-1, v, work.Off(n), a, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Dorm2r(Left, NoTrans, n, n, n-1, u, work, b, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Dorm2r(Right, Trans, n, n, n-1, v, work.Off(n), b, work.Off(2*n)); err != nil {
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

			anorm = golapack.Dlange('1', n, n, a, work)
			bnorm = golapack.Dlange('1', n, n, b, work)

		label100:
			;

			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label110:
			;

			//           Call Dgeqr2, Dorm2r, and Dgghrd to compute H, T, U, and V
			golapack.Dlacpy(Full, n, n, a, h)
			golapack.Dlacpy(Full, n, n, b, t)
			ntest = 1
			result.Set(0, ulpinv)

			if err = golapack.Dgeqr2(n, n, t, work, work.Off(n)); err != nil {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeqr2", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			if err = golapack.Dorm2r(Left, Trans, n, n, n, t, work, h, work.Off(n)); err != nil {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dorm2r", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			golapack.Dlaset(Full, n, n, zero, one, u)
			if err = golapack.Dorm2r(Right, NoTrans, n, n, n, t, work, u, work.Off(n)); err != nil {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dorm2r", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			if err = golapack.Dgghrd('V', 'I', n, 1, n, h, t, u, v); err != nil {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgghrd", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}
			ntest = 4

			//           Do tests 1--4
			result.Set(0, dget51(1, n, a, h, u, v, work))
			result.Set(1, dget51(1, n, b, t, u, v, work))
			result.Set(2, dget51(3, n, b, t, u, u, work))
			result.Set(3, dget51(3, n, b, t, v, v, work))

			//           Call Dhgeqz to compute S1, P1, S2, P2, Q, and Z, do tests.
			//
			//           Compute T1 and UZ
			//
			//           Eigenvalues only
			golapack.Dlacpy(Full, n, n, h, s2)
			golapack.Dlacpy(Full, n, n, t, p2)
			ntest = 5
			result.Set(4, ulpinv)

			if iinfo, err = golapack.Dhgeqz('E', 'N', 'N', n, 1, n, s2, p2, alphr3, alphi3, beta3, q, z, work, lwork); iinfo != 0 || err != nil {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dhgeqz(E)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			//           Eigenvalues and Full Schur Form
			golapack.Dlacpy(Full, n, n, h, s2)
			golapack.Dlacpy(Full, n, n, t, p2)
			//
			if iinfo, err = golapack.Dhgeqz('S', 'N', 'N', n, 1, n, s2, p2, alphr1, alphi1, beta1, q, z, work, lwork); iinfo != 0 || err != nil {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dhgeqz(S)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			//           Eigenvalues, Schur Form, and Schur Vectors
			golapack.Dlacpy(Full, n, n, h, s1)
			golapack.Dlacpy(Full, n, n, t, p1)

			if iinfo, err = golapack.Dhgeqz('S', 'I', 'I', n, 1, n, s1, p1, alphr1, alphi1, beta1, q, z, work, lwork); iinfo != 0 || err != nil {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dhgeqz(V)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			ntest = 8

			//           Do Tests 5--8
			result.Set(4, dget51(1, n, h, s1, q, z, work))
			result.Set(5, dget51(1, n, t, p1, q, z, work))
			result.Set(6, dget51(3, n, t, p1, q, q, work))
			result.Set(7, dget51(3, n, t, p1, z, z, work))

			//           Compute the Left and Right Eigenvectors of (S1,P1)
			//
			//           9: Compute the left eigenvector Matrix without
			//              back transforming:
			ntest = 9
			result.Set(8, ulpinv)

			//           To test "SELECT" option, compute half of the eigenvectors
			//           in one call, and half in another
			i1 = n / 2
			for j = 1; j <= i1; j++ {
				llwork[j-1] = true
			}
			for j = i1 + 1; j <= n; j++ {
				llwork[j-1] = false
			}

			if in, iinfo, err = golapack.Dtgevc(Left, 'S', llwork, n, s1, p1, evectl, dumma.Matrix(u.Rows, opts), n, work); err != nil || iinfo != 0 {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dtgevc(L,S1)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			i1 = in
			for j = 1; j <= i1; j++ {
				llwork[j-1] = false
			}
			for j = i1 + 1; j <= n; j++ {
				llwork[j-1] = true
			}

			if in, iinfo, err = golapack.Dtgevc(Left, 'S', llwork, n, s1, p1, evectl.Off(0, i1), dumma.Matrix(u.Rows, opts), n, work); err != nil || iinfo != 0 {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dtgevc(L,S2)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			dget52(true, n, s1, p1, evectl, alphr1, alphi1, beta1, work, dumma)
			result.Set(8, dumma.Get(0))
			if dumma.Get(1) > thrshn {
				_t.Fail()
				fmt.Printf(" dchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error= %10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Dtgevc(howmny=S)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           10: Compute the left eigenvector Matrix with
			//               back transforming:
			ntest = 10
			result.Set(9, ulpinv)
			golapack.Dlacpy(Full, n, n, q, evectl)
			if in, iinfo, err = golapack.Dtgevc(Left, 'B', llwork, n, s1, p1, evectl, dumma.Matrix(u.Rows, opts), n, work); err != nil || iinfo != 0 {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dtgevc(L,B)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			dget52(true, n, h, t, evectl, alphr1, alphi1, beta1, work, dumma)
			result.Set(9, dumma.Get(0))
			if dumma.Get(1) > thrshn {
				_t.Fail()
				fmt.Printf(" dchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error= %10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Dtgevc(howmny=B)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           11: Compute the right eigenvector Matrix without
			//               back transforming:
			ntest = 11
			result.Set(10, ulpinv)

			//           To test "SELECT" option, compute half of the eigenvectors
			//           in one call, and half in another
			i1 = n / 2
			for j = 1; j <= i1; j++ {
				llwork[j-1] = true
			}
			for j = i1 + 1; j <= n; j++ {
				llwork[j-1] = false
			}

			if in, iinfo, err = golapack.Dtgevc(Right, 'S', llwork, n, s1, p1, dumma.Matrix(u.Rows, opts), evectr, n, work); err != nil || iinfo != 0 {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dtgevc(R,S1)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			i1 = in
			for j = 1; j <= i1; j++ {
				llwork[j-1] = false
			}
			for j = i1 + 1; j <= n; j++ {
				llwork[j-1] = true
			}

			if in, iinfo, err = golapack.Dtgevc(Right, 'S', llwork, n, s1, p1, dumma.Matrix(u.Rows, opts), evectr.Off(0, i1), n, work); err != nil || iinfo != 0 {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dtgevc(R,S2)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			dget52(false, n, s1, p1, evectr, alphr1, alphi1, beta1, work, dumma)
			result.Set(10, dumma.Get(0))
			if dumma.Get(1) > thresh {
				_t.Fail()
				fmt.Printf(" dchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error= %10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Dtgevc(howmny=S)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           12: Compute the right eigenvector Matrix with
			//               back transforming:
			ntest = 12
			result.Set(11, ulpinv)
			golapack.Dlacpy(Full, n, n, z, evectr)
			if in, iinfo, err = golapack.Dtgevc(Right, 'B', llwork, n, s1, p1, dumma.Matrix(u.Rows, opts), evectr, n, work); err != nil || iinfo != 0 {
				_t.Fail()
				fmt.Printf(" dchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dtgevc(R,B)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label210
			}

			dget52(false, n, h, t, evectr, alphr1, alphi1, beta1, work, dumma)
			result.Set(11, dumma.Get(0))
			if dumma.Get(1) > thresh {
				_t.Fail()
				fmt.Printf(" dchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error= %10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Dtgevc(howmny=B)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           Tests 13--15 are done only on request
			if tstdif {
				//              Do Tests 13--14
				result.Set(12, dget51(2, n, s1, s2, q, z, work))
				result.Set(13, dget51(2, n, p1, p2, q, z, work))

				//              Do Test 15
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Abs(alphr1.Get(j-1)-alphr3.Get(j-1))+math.Abs(alphi1.Get(j-1)-alphi3.Get(j-1)))
					temp2 = math.Max(temp2, math.Abs(beta1.Get(j-1)-beta3.Get(j-1)))
				}

				temp1 = temp1 / math.Max(safmin, ulp*math.Max(temp1, anorm))
				temp2 = temp2 / math.Max(safmin, ulp*math.Max(temp2, bnorm))
				result.Set(14, math.Max(temp1, temp2))
				ntest = 15
			} else {
				result.Set(12, zero)
				result.Set(13, zero)
				result.Set(14, zero)
				ntest = 12
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label210:
			;

			ntestt = ntestt + ntest

			//           Print out tests which fail.
			for jr = 1; jr <= ntest; jr++ {
				if result.Get(jr-1) >= thresh {
					_t.Fail()
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Real Generalized eigenvalue problem\n", "DGG")

						//                    Matrix types
						fmt.Printf(" Matrix types (see dchkgg for details): \n")
						fmt.Printf(" Special Matrices:                       (J'=transposed Jordan block)\n   1=(0,0)  2=(I,0)  3=(0,I)  4=(I,I)  5=(J',J')  6=(diag(J',I), diag(I,J'))\n Diagonal Matrices:  ( D=diag(0,1,2,...) )\n   7=(D,I)   9=(large*D, small*I)  11=(large*I, small*D)  13=(large*D, large*I)\n   8=(I,D)  10=(small*D, large*I)  12=(small*I, large*D)  14=(small*D, small*I)\n  15=(D, reversed D)\n")
						fmt.Printf(" Matrices Rotated by Random %s Matrices U, V:\n  16=Transposed Jordan Blocks             19=geometric alpha, beta=0,1\n  17=arithm. alpha&beta                   20=arithmetic alpha, beta=0,1\n  18=clustered alpha, beta=0,1            21=random alpha, beta=0,1\n Large & Small Matrices:\n  22=(large, small)   23=(small,large)    24=(small,small)    25=(large,large)\n  26=random O(1) matrices.\n", "Orthogonal")

						//                    Tests performed
						fmt.Printf("\n Tests performed:   (H is Hessenberg, S is Schur, B, T, P are triangular,\n                    U, V, Q, and Z are %s, l and r are the\n                    appropriate left and right eigenvectors, resp., a is\n                    alpha, b is beta, and %s means %s.)\n 1 = | A - U H V%s | / ( |A| n ulp )      2 = | B - U T V%s | / ( |B| n ulp )\n 3 = | I - UU%s | / ( n ulp )             4 = | I - VV%s | / ( n ulp )\n 5 = | H - Q S Z%s | / ( |H| n ulp )      6 = | T - Q P Z%s | / ( |T| n ulp )\n 7 = | I - QQ%s | / ( n ulp )             8 = | I - ZZ%s | / ( n ulp )\n 9 = math.Max | ( b S - a P )%s l | / const.  10 = math.Max | ( b H - a T )%s l | / const.\n 11= math.Max | ( b S - a P ) r | / const.   12 = math.Max | ( b H - a T ) r | / const.\n \n", "orthogonal", "'", "transpose", "'", "'", "'", "'", "'", "'", "'", "'", "'", "'")

					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, type=%2d, seed=%4d, result %2d is %8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, type=%2d, seed=%4d, result %2d is %10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}

		label230:
		}
	}

	//     Summary
	// dlasum("Dgg", nerrs, ntestt)

	return
}
