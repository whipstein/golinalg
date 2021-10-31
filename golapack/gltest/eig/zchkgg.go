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

// zchkgg checks the nonsymmetric generalized eigenvalue problem
// routines.
//                                H          H        H
// Zgghrd factors A and B as U H V  and U T V , where   means conjugate
// transpose, H is hessenberg, T is triangular and U and V are unitary.
//
//                                 H          H
// Zhgeqz factors H and T as  Q S Z  and Q P Z , where P and S are upper
// triangular and Q and Z are unitary.  It also computes the generalized
// eigenvalues (alpha(1),beta(1)),...,(alpha(n),beta(n)), where
// alpha(j)=S(j,j) and beta(j)=P(j,j) -- thus, w(j) = alpha(j)/beta(j)
// is a root of the generalized eigenvalue problem
//
//     det( A - w(j) B ) = 0
//
// and m(j) = beta(j)/alpha(j) is a root of the essentially equivalent
// problem
//
//     det( m(j) A - B ) = 0
//
// Ztgevc computes the matrix L of left eigenvectors and the matrix R
// of right eigenvectors for the matrix pair ( S, P ).  In the
// description below,  l and r are left and right eigenvectors
// corresponding to the generalized eigenvalues (alpha,beta).
//
// When zchkgg is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each _type of matrix, one matrix will be generated and used
// to test the nonsymmetric eigenroutines.  For each matrix, 13
// tests will be performed.  The first twelve "test ratios" should be
// small -- O(1).  They will be compared with the threshold THRESH:
//
//                  H
// (1)   | A - U H V  | / ( |A| n ulp )
//
//                  H
// (2)   | B - U T V  | / ( |B| n ulp )
//
//               H
// (3)   | I - UU  | / ( n ulp )
//
//               H
// (4)   | I - VV  | / ( n ulp )
//
//                  H
// (5)   | H - Q S Z  | / ( |H| n ulp )
//
//                  H
// (6)   | T - Q P Z  | / ( |T| n ulp )
//
//               H
// (7)   | I - QQ  | / ( n ulp )
//
//               H
// (8)   | I - ZZ  | / ( n ulp )
//
// (9)   max over all left eigenvalue/-vector pairs (beta/alpha,l) of
//                           H
//       | (beta A - alpha B) l | / ( ulp max( |beta A|, |alpha B| ) )
//
// (10)  max over all left eigenvalue/-vector pairs (beta/alpha,l') of
//                           H
//       | (beta H - alpha T) l' | / ( ulp max( |beta H|, |alpha T| ) )
//
//       where the eigenvectors l' are the result of passing Q to
//       DTGEVC and back transforming (JOB='B').
//
// (11)  max over all right eigenvalue/-vector pairs (beta/alpha,r) of
//
//       | (beta A - alpha B) r | / ( ulp max( |beta A|, |alpha B| ) )
//
// (12)  max over all right eigenvalue/-vector pairs (beta/alpha,r') of
//
//       | (beta H - alpha T) r' | / ( ulp max( |beta H|, |alpha T| ) )
//
//       where the eigenvectors r' are the result of passing Z to
//       DTGEVC and back transforming (JOB='B').
//
// The last three test ratios will usually be small, but there is no
// mathematical requirement that they be so.  They are therefore
// compared with THRESH only if TSTDIF is .TRUE.
//
// (13)  | S(Q,Z computed) - S(Q,Z not computed) | / ( |S| ulp )
//
// (14)  | P(Q,Z computed) - P(Q,Z not computed) | / ( |P| ulp )
//
// (15)  max( |alpha(Q,Z computed) - alpha(Q,Z not computed)|/|S| ,
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
// (7)  ( D, I )         where D is P*D1, P is a random unitary diagonal
//                       matrix (i.e., with random magnitude 1 entries
//                       on the diagonal), and D1=diag( 0, 1,..., N-1 )
//                       (i.e., a diagonal matrix with D1(1,1)=0,
//                       D1(2,2)=1, ..., D1(N,N)=N-1.)
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
// (15) ( D1, D2 )        where D1=P*diag( 0, 0, 1, ..., N-3, 0 ) and
//                        D2=Q*diag( 0, N-3, N-4,..., 1, 0, 0 ), and
//                        P and Q are random unitary diagonal matrices.
//           t   t
// (16) U ( J , J ) V     where U and V are random unitary matrices.
//
// (17) U ( T1, T2 ) V    where T1 and T2 are upper triangular matrices
//                        with random O(1) entries above the diagonal
//                        and diagonal entries diag(T1) =
//                        P*( 0, 0, 1, ..., N-3, 0 ) and diag(T2) =
//                        Q*( 0, N-3, N-4,..., 1, 0, 0 )
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
// (22) U ( big*T1, small*T2 ) V   diag(T1) = P*( 0, 0, 1, ..., N-3, 0 )
//                                 diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (23) U ( small*T1, big*T2 ) V   diag(T1) = P*( 0, 0, 1, ..., N-3, 0 )
//                                 diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (24) U ( small*T1, small*T2 ) V diag(T1) = P*( 0, 0, 1, ..., N-3, 0 )
//                                 diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (25) U ( big*T1, big*T2 ) V     diag(T1) = P*( 0, 0, 1, ..., N-3, 0 )
//                                 diag(T2) = ( 0, 1, ..., 1, 0, 0 )
//
// (26) U ( T1, T2 ) V     where T1 and T2 are random upper-triangular
//                         matrices.
func zchkgg(nsizes int, nn []int, ntypes int, dotype []bool, iseed *[]int, thresh float64, tstdif bool, thrshn float64, a, b, h, t, s1, s2, p1, p2, u, v, q, z *mat.CMatrix, alpha1, beta1, alpha3, beta3 *mat.CVector, evectl, evectr *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, llwork []bool, result *mat.Vector) (nerrs, ntestt int, err error) {
	var badnn bool
	var cone, ctemp, czero complex128
	var anorm, bnorm, one, safmax, safmin, temp1, temp2, ulp, ulpinv, zero float64
	var i1, iadd, iinfo, in, j, jc, jr, jsize, jtype, lwkopt, maxtyp, mtypes, n, n1, nmats, nmax, ntest int
	lasign := []bool{false, false, false, false, false, false, true, false, true, true, false, false, true, true, true, false, true, false, false, false, true, true, true, true, true, false}
	lbsign := []bool{false, false, false, false, false, false, false, true, false, false, true, true, false, false, true, false, true, false, false, false, false, false, false, false, false, false}
	cdumma := cvf(4)
	dumma := vf(4)
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

	lwkopt = max(2*nmax*nmax, 4*nmax, 1)

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsizes=%v", nsizes)
	} else if badnn {
		err = fmt.Errorf("badnn, nn=%v", nn)
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
		gltest.Xerbla2("zchkgg", err)
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
				ioldsd[j-1] = (*iseed)[j-1]
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
			//           RMAGN:  used to implement KAMAGN and KBMAGN.
			if mtypes > maxtyp {
				goto label110
			}
			iinfo = 0
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
				zlatm4(katype[jtype-1], in, kz1[kazero[jtype-1]-1], kz2[kazero[jtype-1]-1], lasign[jtype-1], rmagn.Get(kamagn[jtype-1]-0), ulp, rmagn.Get(ktrian[jtype-1]*kamagn[jtype-1]-0), 4, iseed, a)
				iadd = kadd[kazero[jtype-1]-1]
				if iadd > 0 && iadd <= n {
					a.SetRe(iadd-1, iadd-1, rmagn.Get(kamagn[jtype-1]-0))
				}
				//
				//              Generate B (w/o rotation)
				//
				if abs(kbtype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Zlaset(Full, n, n, czero, czero, b)
					}
				} else {
					in = n
				}
				zlatm4(kbtype[jtype-1], in, kz1[kbzero[jtype-1]-1], kz2[kbzero[jtype-1]-1], lbsign[jtype-1], rmagn.Get(kbmagn[jtype-1]-0), one, rmagn.Get(ktrian[jtype-1]*kbmagn[jtype-1]-0), 4, iseed, b)
				iadd = kadd[kbzero[jtype-1]-1]
				if iadd != 0 {
					b.SetRe(iadd-1, iadd-1, rmagn.Get(kbmagn[jtype-1]-0))
				}

				if kclass[jtype-1] == 2 && n > 0 {
					//                 Include rotations
					//
					//                 Generate U, V as Householder transformations times a
					//                 diagonal matrix.  (Note that ZLARFG makes U(j,j) and
					//                 V(j,j) real.)
					for jc = 1; jc <= n-1; jc++ {
						for jr = jc; jr <= n; jr++ {
							u.Set(jr-1, jc-1, matgen.Zlarnd(3, *iseed))
							v.Set(jr-1, jc-1, matgen.Zlarnd(3, *iseed))
						}
						*u.GetPtr(jc-1, jc-1), *work.GetPtr(jc - 1) = golapack.Zlarfg(n+1-jc, u.Get(jc-1, jc-1), u.CVector(jc, jc-1, 1))
						work.SetRe(2*n+jc-1, math.Copysign(one, u.GetRe(jc-1, jc-1)))
						u.Set(jc-1, jc-1, cone)
						*v.GetPtr(jc-1, jc-1), *work.GetPtr(n + jc - 1) = golapack.Zlarfg(n+1-jc, v.Get(jc-1, jc-1), v.CVector(jc, jc-1, 1))
						work.SetRe(3*n+jc-1, math.Copysign(one, v.GetRe(jc-1, jc-1)))
						v.Set(jc-1, jc-1, cone)
					}
					ctemp = matgen.Zlarnd(3, *iseed)
					u.Set(n-1, n-1, cone)
					work.Set(n-1, czero)
					work.Set(3*n-1, ctemp/complex(cmplx.Abs(ctemp), 0))
					ctemp = matgen.Zlarnd(3, *iseed)
					v.Set(n-1, n-1, cone)
					work.Set(2*n-1, czero)
					work.Set(4*n-1, ctemp/complex(cmplx.Abs(ctemp), 0))

					//                 Apply the diagonal matrices
					for jc = 1; jc <= n; jc++ {
						for jr = 1; jr <= n; jr++ {
							a.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.GetConj(3*n+jc-1)*a.Get(jr-1, jc-1))
							b.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.GetConj(3*n+jc-1)*b.Get(jr-1, jc-1))
						}
					}
					if err = golapack.Zunm2r(Left, NoTrans, n, n, n-1, u, work, a, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Zunm2r(Right, ConjTrans, n, n, n-1, v, work.Off(n), a, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Zunm2r(Left, NoTrans, n, n, n-1, u, work, b, work.Off(2*n)); err != nil {
						goto label100
					}
					if err = golapack.Zunm2r(Right, ConjTrans, n, n, n-1, v, work.Off(n), b, work.Off(2*n)); err != nil {
						goto label100
					}
				}
			} else {
				//              Random matrices
				for jc = 1; jc <= n; jc++ {
					for jr = 1; jr <= n; jr++ {
						a.Set(jr-1, jc-1, rmagn.GetCmplx(kamagn[jtype-1]-0)*matgen.Zlarnd(4, *iseed))
						b.Set(jr-1, jc-1, rmagn.GetCmplx(kbmagn[jtype-1]-0)*matgen.Zlarnd(4, *iseed))
					}
				}
			}

			anorm = golapack.Zlange('1', n, n, a, rwork)
			bnorm = golapack.Zlange('1', n, n, b, rwork)

		label100:
			;

			if iinfo != 0 {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label110:
			;

			//           Call Zgeqr2, Zunm2r, and Zgghrd to compute H, T, U, and V
			golapack.Zlacpy(Full, n, n, a, h)
			golapack.Zlacpy(Full, n, n, b, t)
			ntest = 1
			result.Set(0, ulpinv)

			if err = golapack.Zgeqr2(n, n, t, work, work.Off(n)); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zgeqr2", iinfo, n, jtype, ioldsd)
				goto label210
			}

			if err = golapack.Zunm2r(Left, ConjTrans, n, n, n, t, work, h, work.Off(n)); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zunm2r", iinfo, n, jtype, ioldsd)
				goto label210
			}

			golapack.Zlaset(Full, n, n, czero, cone, u)
			if err = golapack.Zunm2r(Right, NoTrans, n, n, n, t, work, u, work.Off(n)); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zunm2r", iinfo, n, jtype, ioldsd)
				goto label210
			}

			if err = golapack.Zgghrd('V', 'I', n, 1, n, h, t, u, v); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zgghrd", iinfo, n, jtype, ioldsd)
				goto label210
			}
			ntest = 4

			//           Do tests 1--4
			result.Set(0, zget51(1, n, a, h, u, v, work, rwork))
			result.Set(1, zget51(1, n, b, t, u, v, work, rwork))
			result.Set(2, zget51(3, n, b, t, u, u, work, rwork))
			result.Set(3, zget51(3, n, b, t, v, v, work, rwork))

			//           Call Zhgeqz to compute S1, P1, S2, P2, Q, and Z, do tests.
			//
			//           Compute T1 and UZ
			//
			//           Eigenvalues only
			golapack.Zlacpy(Full, n, n, h, s2)
			golapack.Zlacpy(Full, n, n, t, p2)
			ntest = 5
			result.Set(4, ulpinv)

			if iinfo, err = golapack.Zhgeqz('E', 'N', 'N', n, 1, n, s2, p2, alpha3, beta3, q, z, work, lwork, rwork); err != nil || iinfo != 0 {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhgeqz(E)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			//           Eigenvalues and Full Schur Form
			golapack.Zlacpy(Full, n, n, h, s2)
			golapack.Zlacpy(Full, n, n, t, p2)

			if iinfo, err = golapack.Zhgeqz('S', 'N', 'N', n, 1, n, s2, p2, alpha1, beta1, q, z, work, lwork, rwork); err != nil || iinfo != 0 {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhgeqz(S)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			//           Eigenvalues, Schur Form, and Schur Vectors
			golapack.Zlacpy(Full, n, n, h, s1)
			golapack.Zlacpy(Full, n, n, t, p1)

			if iinfo, err = golapack.Zhgeqz('S', 'I', 'I', n, 1, n, s1, p1, alpha1, beta1, q, z, work, lwork, rwork); err != nil || iinfo != 0 {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhgeqz(V)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			ntest = 8

			//           Do Tests 5--8
			result.Set(4, zget51(1, n, h, s1, q, z, work, rwork))
			result.Set(5, zget51(1, n, t, p1, q, z, work, rwork))
			result.Set(6, zget51(3, n, t, p1, q, q, work, rwork))
			result.Set(7, zget51(3, n, t, p1, z, z, work, rwork))

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

			if in, err = golapack.Ztgevc(Left, 'S', llwork, n, s1, p1, evectl, cdumma.CMatrix(u.Rows, opts), n, work, rwork); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztgevc(L,S1)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			i1 = in
			for j = 1; j <= i1; j++ {
				llwork[j-1] = false
			}
			for j = i1 + 1; j <= n; j++ {
				llwork[j-1] = true
			}

			if in, err = golapack.Ztgevc(Left, 'S', llwork, n, s1, p1, evectl.Off(0, i1), cdumma.CMatrix(u.Rows, opts), n, work, rwork); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztgevc(L,S2)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			zget52(true, n, s1, p1, evectl, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(8, dumma.Get(0))
			if dumma.Get(1) > thrshn {
				fmt.Printf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Ztgevc(HOWMNY=S)", dumma.Get(1), n, jtype, ioldsd)
				err = fmt.Errorf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Ztgevc(HOWMNY=S)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           10: Compute the left eigenvector Matrix with
			//               back transforming:
			ntest = 10
			result.Set(9, ulpinv)
			golapack.Zlacpy(Full, n, n, q, evectl)
			if in, err = golapack.Ztgevc(Left, 'B', llwork, n, s1, p1, evectl, cdumma.CMatrix(u.Rows, opts), n, work, rwork); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztgevc(L,B)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			zget52(true, n, h, t, evectl, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(9, dumma.Get(0))
			if dumma.Get(1) > thrshn {
				fmt.Printf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Ztgevc(HOWMNY=B)", dumma.Get(1), n, jtype, ioldsd)
				err = fmt.Errorf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Ztgevc(HOWMNY=B)", dumma.Get(1), n, jtype, ioldsd)
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

			if in, err = golapack.Ztgevc(Right, 'S', llwork, n, s1, p1, cdumma.CMatrix(u.Rows, opts), evectr, n, work, rwork); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztgevc(R,S1)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			i1 = in
			for j = 1; j <= i1; j++ {
				llwork[j-1] = false
			}
			for j = i1 + 1; j <= n; j++ {
				llwork[j-1] = true
			}

			if in, err = golapack.Ztgevc(Right, 'S', llwork, n, s1, p1, cdumma.CMatrix(u.Rows, opts), evectr.Off(0, i1), n, work, rwork); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztgevc(R,S2)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			zget52(false, n, s1, p1, evectr, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(10, dumma.Get(0))
			if dumma.Get(1) > thresh {
				fmt.Printf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Ztgevc(HOWMNY=S)", dumma.Get(1), n, jtype, ioldsd)
				err = fmt.Errorf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Ztgevc(HOWMNY=S)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           12: Compute the right eigenvector Matrix with
			//               back transforming:
			ntest = 12
			result.Set(11, ulpinv)
			golapack.Zlacpy(Full, n, n, z, evectr)
			if in, err = golapack.Ztgevc(Right, 'B', llwork, n, s1, p1, cdumma.CMatrix(u.Rows, opts), evectr, n, work, rwork); err != nil {
				fmt.Printf(" zchkgg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztgevc(R,B)", iinfo, n, jtype, ioldsd)
				goto label210
			}

			zget52(false, n, h, t, evectr, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(11, dumma.Get(0))
			if dumma.Get(1) > thresh {
				fmt.Printf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Ztgevc(HOWMNY=B)", dumma.Get(1), n, jtype, ioldsd)
				err = fmt.Errorf(" zchkgg: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Ztgevc(HOWMNY=B)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           Tests 13--15 are done only on request
			if tstdif {
				//              Do Tests 13--14
				result.Set(12, zget51(2, n, s1, s2, q, z, work, rwork))
				result.Set(13, zget51(2, n, p1, p2, q, z, work, rwork))

				//              Do Test 15
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, cmplx.Abs(alpha1.Get(j-1)-alpha3.Get(j-1)))
					temp2 = math.Max(temp2, cmplx.Abs(beta1.Get(j-1)-beta3.Get(j-1)))
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
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf(" %3s -- Complex Generalized eigenvalue problem\n", "ZGG")

						//                    Matrix types
						fmt.Printf(" Matrix types (see zchkgg for details): \n")
						fmt.Printf(" Special Matrices:                       (J'=transposed Jordan block)\n   1=(0,0)  2=(I,0)  3=(0,I)  4=(I,I)  5=(J',J')  6=(diag(J',I), diag(I,J'))\n Diagonal Matrices:  ( D=diag(0,1,2,...) )\n   7=(D,I)   9=(large*D, small*I)  11=(large*I, small*D)  13=(large*D, large*I)\n   8=(I,D)  10=(small*D, large*I)  12=(small*I, large*D)  14=(small*D, small*I)\n  15=(D, reversed D)\n")
						fmt.Printf(" Matrices Rotated by Random %s Matrices U, V:\n  16=Transposed Jordan Blocks             19=geometric alpha, beta=0,1\n  17=arithm. alpha&beta                   20=arithmetic alpha, beta=0,1\n  18=clustered alpha, beta=0,1            21=random alpha, beta=0,1\n Large & Small Matrices:\n  22=(large, small)   23=(small,large)    24=(small,small)    25=(large,large)\n  26=random O(1) matrices.\n", "Unitary")

						//                    Tests performed
						fmt.Printf("\n Tests performed:   (H is Hessenberg, S is Schur, B, T, P are triangular,\n                    U, V, Q, and Z are %s, l and r are the\n                    appropriate left and right eigenvectors, resp., a is\n                    alpha, b is beta, and %s means %s.)\n 1 = | A - U H V%s | / ( |A| n ulp )      2 = | B - U T V%s | / ( |B| n ulp )\n 3 = | I - UU%s | / ( n ulp )             4 = | I - VV%s | / ( n ulp )\n 5 = | H - Q S Z%s | / ( |H| n ulp )      6 = | T - Q P Z%s | / ( |T| n ulp )\n 7 = | I - QQ%s | / ( n ulp )             8 = | I - ZZ%s | / ( n ulp )\n 9 = max | ( b S - a P )%s l | / const.  10 = max | ( b H - a T )%s l | / const.\n 11= max | ( b S - a P ) r | / const.   12 = max | ( b H - a T ) r | / const.\n \n", "unitary", "*", "conjugate transpose", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*")

					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
						err = fmt.Errorf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
						err = fmt.Errorf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}

		label230:
		}
	}

	//     Summary
	// dlasum("Zgg", nerrs, ntestt)

	return
}
