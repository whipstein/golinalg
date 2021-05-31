package eig

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkgg checks the nonsymmetric generalized eigenvalue problem
// routines.
//                                H          H        H
// ZGGHRD factors A and B as U H V  and U T V , where   means conjugate
// transpose, H is hessenberg, T is triangular and U and V are unitary.
//
//                                 H          H
// ZHGEQZ factors H and T as  Q S Z  and Q P Z , where P and S are upper
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
// ZTGEVC computes the matrix L of left eigenvectors and the matrix R
// of right eigenvectors for the matrix pair ( S, P ).  In the
// description below,  l and r are left and right eigenvectors
// corresponding to the generalized eigenvalues (alpha,beta).
//
// When ZCHKGG is called, a number of matrix "sizes" ("n's") and a
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
// (9)   maxint over all left eigenvalue/-vector pairs (beta/alpha,l) of
//                           H
//       | (beta A - alpha B) l | / ( ulp maxint( |beta A|, |alpha B| ) )
//
// (10)  maxint over all left eigenvalue/-vector pairs (beta/alpha,l') of
//                           H
//       | (beta H - alpha T) l' | / ( ulp maxint( |beta H|, |alpha T| ) )
//
//       where the eigenvectors l' are the result of passing Q to
//       DTGEVC and back transforming (JOB='B').
//
// (11)  maxint over all right eigenvalue/-vector pairs (beta/alpha,r) of
//
//       | (beta A - alpha B) r | / ( ulp maxint( |beta A|, |alpha B| ) )
//
// (12)  maxint over all right eigenvalue/-vector pairs (beta/alpha,r') of
//
//       | (beta H - alpha T) r' | / ( ulp maxint( |beta H|, |alpha T| ) )
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
// (15)  maxint( |alpha(Q,Z computed) - alpha(Q,Z not computed)|/|S| ,
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
func Zchkgg(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, tstdif *bool, thrshn *float64, nounit *int, a *mat.CMatrix, lda *int, b, h, t, s1, s2, p1, p2, u *mat.CMatrix, ldu *int, v, q, z *mat.CMatrix, alpha1, beta1, alpha3, beta3 *mat.CVector, evectl, evectr *mat.CMatrix, work *mat.CVector, lwork *int, rwork *mat.Vector, llwork *[]bool, result *mat.Vector, info *int, _t *testing.T) {
	var badnn bool
	var cone, ctemp, czero complex128
	var anorm, bnorm, one, safmax, safmin, temp1, temp2, ulp, ulpinv, zero float64
	var i1, iadd, iinfo, in, j, jc, jr, jsize, jtype, lwkopt, maxtyp, mtypes, n, n1, nerrs, nmats, nmax, ntest, ntestt int
	lasign := make([]bool, 26)
	lbsign := make([]bool, 26)
	cdumma := cvf(4)
	dumma := vf(4)
	rmagn := vf(4)
	ioldsd := make([]int, 4)
	kadd := make([]int, 6)
	kamagn := make([]int, 26)
	katype := make([]int, 26)
	kazero := make([]int, 26)
	kbmagn := make([]int, 26)
	kbtype := make([]int, 26)
	kbzero := make([]int, 26)
	kclass := make([]int, 26)
	ktrian := make([]int, 26)
	kz1 := make([]int, 6)
	kz2 := make([]int, 6)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 26

	kclass[0], kclass[1], kclass[2], kclass[3], kclass[4], kclass[5], kclass[6], kclass[7], kclass[8], kclass[9], kclass[10], kclass[11], kclass[12], kclass[13], kclass[14], kclass[15], kclass[16], kclass[17], kclass[18], kclass[19], kclass[20], kclass[21], kclass[22], kclass[23], kclass[24], kclass[25] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3
	kz1[0], kz1[1], kz1[2], kz1[3], kz1[4], kz1[5] = 0, 1, 2, 1, 3, 3
	kz2[0], kz2[1], kz2[2], kz2[3], kz2[4], kz2[5] = 0, 0, 1, 2, 1, 1
	kadd[0], kadd[1], kadd[2], kadd[3], kadd[4], kadd[5] = 0, 0, 0, 0, 3, 2
	katype[0], katype[1], katype[2], katype[3], katype[4], katype[5], katype[6], katype[7], katype[8], katype[9], katype[10], katype[11], katype[12], katype[13], katype[14], katype[15], katype[16], katype[17], katype[18], katype[19], katype[20], katype[21], katype[22], katype[23], katype[24], katype[25] = 0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4, 4, 4, 2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0
	kbtype[0], kbtype[1], kbtype[2], kbtype[3], kbtype[4], kbtype[5], kbtype[6], kbtype[7], kbtype[8], kbtype[9], kbtype[10], kbtype[11], kbtype[12], kbtype[13], kbtype[14], kbtype[15], kbtype[16], kbtype[17], kbtype[18], kbtype[19], kbtype[20], kbtype[21], kbtype[22], kbtype[23], kbtype[24], kbtype[25] = 0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4, 1, 1, -4, 2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0
	kazero[0], kazero[1], kazero[2], kazero[3], kazero[4], kazero[5], kazero[6], kazero[7], kazero[8], kazero[9], kazero[10], kazero[11], kazero[12], kazero[13], kazero[14], kazero[15], kazero[16], kazero[17], kazero[18], kazero[19], kazero[20], kazero[21], kazero[22], kazero[23], kazero[24], kazero[25] = 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3, 1, 3, 5, 5, 5, 5, 3, 3, 3, 3, 1
	kbzero[0], kbzero[1], kbzero[2], kbzero[3], kbzero[4], kbzero[5], kbzero[6], kbzero[7], kbzero[8], kbzero[9], kbzero[10], kbzero[11], kbzero[12], kbzero[13], kbzero[14], kbzero[15], kbzero[16], kbzero[17], kbzero[18], kbzero[19], kbzero[20], kbzero[21], kbzero[22], kbzero[23], kbzero[24], kbzero[25] = 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4, 1, 4, 6, 6, 6, 6, 4, 4, 4, 4, 1
	kamagn[0], kamagn[1], kamagn[2], kamagn[3], kamagn[4], kamagn[5], kamagn[6], kamagn[7], kamagn[8], kamagn[9], kamagn[10], kamagn[11], kamagn[12], kamagn[13], kamagn[14], kamagn[15], kamagn[16], kamagn[17], kamagn[18], kamagn[19], kamagn[20], kamagn[21], kamagn[22], kamagn[23], kamagn[24], kamagn[25] = 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1
	kbmagn[0], kbmagn[1], kbmagn[2], kbmagn[3], kbmagn[4], kbmagn[5], kbmagn[6], kbmagn[7], kbmagn[8], kbmagn[9], kbmagn[10], kbmagn[11], kbmagn[12], kbmagn[13], kbmagn[14], kbmagn[15], kbmagn[16], kbmagn[17], kbmagn[18], kbmagn[19], kbmagn[20], kbmagn[21], kbmagn[22], kbmagn[23], kbmagn[24], kbmagn[25] = 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1
	ktrian[0], ktrian[1], ktrian[2], ktrian[3], ktrian[4], ktrian[5], ktrian[6], ktrian[7], ktrian[8], ktrian[9], ktrian[10], ktrian[11], ktrian[12], ktrian[13], ktrian[14], ktrian[15], ktrian[16], ktrian[17], ktrian[18], ktrian[19], ktrian[20], ktrian[21], ktrian[22], ktrian[23], ktrian[24], ktrian[25] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	lasign[0], lasign[1], lasign[2], lasign[3], lasign[4], lasign[5], lasign[6], lasign[7], lasign[8], lasign[9], lasign[10], lasign[11], lasign[12], lasign[13], lasign[14], lasign[15], lasign[16], lasign[17], lasign[18], lasign[19], lasign[20], lasign[21], lasign[22], lasign[23], lasign[24], lasign[25] = false, false, false, false, false, false, true, false, true, true, false, false, true, true, true, false, true, false, false, false, true, true, true, true, true, false
	lbsign[0], lbsign[1], lbsign[2], lbsign[3], lbsign[4], lbsign[5], lbsign[6], lbsign[7], lbsign[8], lbsign[9], lbsign[10], lbsign[11], lbsign[12], lbsign[13], lbsign[14], lbsign[15], lbsign[16], lbsign[17], lbsign[18], lbsign[19], lbsign[20], lbsign[21], lbsign[22], lbsign[23], lbsign[24], lbsign[25] = false, false, false, false, false, false, false, true, false, false, true, true, false, false, true, false, true, false, false, false, false, false, false, false, false, false

	//     Check for errors
	(*info) = 0

	badnn = false
	nmax = 1
	for j = 1; j <= (*nsizes); j++ {
		nmax = maxint(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
	}

	lwkopt = maxint(2*nmax*nmax, 4*nmax, 1)

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badnn {
		(*info) = -2
	} else if (*ntypes) < 0 {
		(*info) = -3
	} else if (*thresh) < zero {
		(*info) = -6
	} else if (*lda) <= 1 || (*lda) < nmax {
		(*info) = -10
	} else if (*ldu) <= 1 || (*ldu) < nmax {
		(*info) = -19
	} else if lwkopt > (*lwork) {
		(*info) = -30
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZCHKGG"), -(*info))
		return
	}

	//     Quick return if possible
	if (*nsizes) == 0 || (*ntypes) == 0 {
		return
	}

	safmin = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	safmin = safmin / ulp
	safmax = one / safmin
	golapack.Dlabad(&safmin, &safmax)
	ulpinv = one / ulp

	//     The values RMAGN(2:3) depend on N, see below.
	rmagn.Set(0, zero)
	rmagn.Set(1, one)

	//     Loop over sizes, types
	ntestt = 0
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		n = (*nn)[jsize-1]
		n1 = maxint(1, n)
		rmagn.Set(2, safmax*ulp/float64(n1))
		rmagn.Set(3, safmin*ulpinv*float64(n1))

		if (*nsizes) != 1 {
			mtypes = minint(maxtyp, *ntypes)
		} else {
			mtypes = minint(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label230
			}
			nmats = nmats + 1
			ntest = 0

			//           Save ISEED in case of an error.
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
				if absint(katype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Zlaset('F', &n, &n, &czero, &czero, a, lda)
					}
				} else {
					in = n
				}
				Zlatm4(&katype[jtype-1], &in, &kz1[kazero[jtype-1]-1], &kz2[kazero[jtype-1]-1], lasign[jtype-1], rmagn.GetPtr(kamagn[jtype-1]-0), &ulp, rmagn.GetPtr(ktrian[jtype-1]*kamagn[jtype-1]-0), func() *int { y := 4; return &y }(), iseed, a, lda)
				iadd = kadd[kazero[jtype-1]-1]
				if iadd > 0 && iadd <= n {
					a.SetRe(iadd-1, iadd-1, rmagn.Get(kamagn[jtype-1]-0))
				}
				//
				//              Generate B (w/o rotation)
				//
				if absint(kbtype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Zlaset('F', &n, &n, &czero, &czero, b, lda)
					}
				} else {
					in = n
				}
				Zlatm4(&kbtype[jtype-1], &in, &kz1[kbzero[jtype-1]-1], &kz2[kbzero[jtype-1]-1], lbsign[jtype-1], rmagn.GetPtr(kbmagn[jtype-1]-0), &one, rmagn.GetPtr(ktrian[jtype-1]*kbmagn[jtype-1]-0), func() *int { y := 4; return &y }(), iseed, b, lda)
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
							u.Set(jr-1, jc-1, matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed))
							v.Set(jr-1, jc-1, matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed))
						}
						golapack.Zlarfg(toPtr(n+1-jc), u.GetPtr(jc-1, jc-1), u.CVector(jc+1-1, jc-1), func() *int { y := 1; return &y }(), work.GetPtr(jc-1))
						work.SetRe(2*n+jc-1, math.Copysign(one, u.GetRe(jc-1, jc-1)))
						u.Set(jc-1, jc-1, cone)
						golapack.Zlarfg(toPtr(n+1-jc), v.GetPtr(jc-1, jc-1), v.CVector(jc+1-1, jc-1), func() *int { y := 1; return &y }(), work.GetPtr(n+jc-1))
						work.SetRe(3*n+jc-1, math.Copysign(one, v.GetRe(jc-1, jc-1)))
						v.Set(jc-1, jc-1, cone)
					}
					ctemp = matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed)
					u.Set(n-1, n-1, cone)
					work.Set(n-1, czero)
					work.Set(3*n-1, ctemp/complex(cmplx.Abs(ctemp), 0))
					ctemp = matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed)
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
					golapack.Zunm2r('L', 'N', &n, &n, toPtr(n-1), u, ldu, work, a, lda, work.Off(2*n+1-1), &iinfo)
					if iinfo != 0 {
						goto label100
					}
					golapack.Zunm2r('R', 'C', &n, &n, toPtr(n-1), v, ldu, work.Off(n+1-1), a, lda, work.Off(2*n+1-1), &iinfo)
					if iinfo != 0 {
						goto label100
					}
					golapack.Zunm2r('L', 'N', &n, &n, toPtr(n-1), u, ldu, work, b, lda, work.Off(2*n+1-1), &iinfo)
					if iinfo != 0 {
						goto label100
					}
					golapack.Zunm2r('R', 'C', &n, &n, toPtr(n-1), v, ldu, work.Off(n+1-1), b, lda, work.Off(2*n+1-1), &iinfo)
					if iinfo != 0 {
						goto label100
					}
				}
			} else {
				//              Random matrices
				for jc = 1; jc <= n; jc++ {
					for jr = 1; jr <= n; jr++ {
						a.Set(jr-1, jc-1, rmagn.GetCmplx(kamagn[jtype-1]-0)*matgen.Zlarnd(func() *int { y := 4; return &y }(), iseed))
						b.Set(jr-1, jc-1, rmagn.GetCmplx(kbmagn[jtype-1]-0)*matgen.Zlarnd(func() *int { y := 4; return &y }(), iseed))
					}
				}
			}

			anorm = golapack.Zlange('1', &n, &n, a, lda, rwork)
			bnorm = golapack.Zlange('1', &n, &n, b, lda, rwork)

		label100:
			;

			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				return
			}

		label110:
			;

			//           Call ZGEQR2, ZUNM2R, and ZGGHRD to compute H, T, U, and V
			golapack.Zlacpy(' ', &n, &n, a, lda, h, lda)
			golapack.Zlacpy(' ', &n, &n, b, lda, t, lda)
			ntest = 1
			result.Set(0, ulpinv)

			golapack.Zgeqr2(&n, &n, t, lda, work, work.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEQR2", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			golapack.Zunm2r('L', 'C', &n, &n, &n, t, lda, work, h, lda, work.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUNM2R", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			golapack.Zlaset('F', &n, &n, &czero, &cone, u, ldu)
			golapack.Zunm2r('R', 'N', &n, &n, &n, t, lda, work, u, ldu, work.Off(n+1-1), &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUNM2R", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			golapack.Zgghrd('V', 'I', &n, func() *int { y := 1; return &y }(), &n, h, lda, t, lda, u, ldu, v, ldu, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGGHRD", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}
			ntest = 4

			//           Do tests 1--4
			Zget51(func() *int { y := 1; return &y }(), &n, a, lda, h, lda, u, ldu, v, ldu, work, rwork, result.GetPtr(0))
			Zget51(func() *int { y := 1; return &y }(), &n, b, lda, t, lda, u, ldu, v, ldu, work, rwork, result.GetPtr(1))
			Zget51(func() *int { y := 3; return &y }(), &n, b, lda, t, lda, u, ldu, u, ldu, work, rwork, result.GetPtr(2))
			Zget51(func() *int { y := 3; return &y }(), &n, b, lda, t, lda, v, ldu, v, ldu, work, rwork, result.GetPtr(3))

			//           Call ZHGEQZ to compute S1, P1, S2, P2, Q, and Z, do tests.
			//
			//           Compute T1 and UZ
			//
			//           Eigenvalues only
			golapack.Zlacpy(' ', &n, &n, h, lda, s2, lda)
			golapack.Zlacpy(' ', &n, &n, t, lda, p2, lda)
			ntest = 5
			result.Set(4, ulpinv)

			golapack.Zhgeqz('E', 'N', 'N', &n, func() *int { y := 1; return &y }(), &n, s2, lda, p2, lda, alpha3, beta3, q, ldu, z, ldu, work, lwork, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHGEQZ(E)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			//           Eigenvalues and Full Schur Form
			golapack.Zlacpy(' ', &n, &n, h, lda, s2, lda)
			golapack.Zlacpy(' ', &n, &n, t, lda, p2, lda)

			golapack.Zhgeqz('S', 'N', 'N', &n, func() *int { y := 1; return &y }(), &n, s2, lda, p2, lda, alpha1, beta1, q, ldu, z, ldu, work, lwork, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHGEQZ(S)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			//           Eigenvalues, Schur Form, and Schur Vectors
			golapack.Zlacpy(' ', &n, &n, h, lda, s1, lda)
			golapack.Zlacpy(' ', &n, &n, t, lda, p1, lda)

			golapack.Zhgeqz('S', 'I', 'I', &n, func() *int { y := 1; return &y }(), &n, s1, lda, p1, lda, alpha1, beta1, q, ldu, z, ldu, work, lwork, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHGEQZ(V)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			ntest = 8

			//           Do Tests 5--8
			Zget51(func() *int { y := 1; return &y }(), &n, h, lda, s1, lda, q, ldu, z, ldu, work, rwork, result.GetPtr(4))
			Zget51(func() *int { y := 1; return &y }(), &n, t, lda, p1, lda, q, ldu, z, ldu, work, rwork, result.GetPtr(5))
			Zget51(func() *int { y := 3; return &y }(), &n, t, lda, p1, lda, q, ldu, q, ldu, work, rwork, result.GetPtr(6))
			Zget51(func() *int { y := 3; return &y }(), &n, t, lda, p1, lda, z, ldu, z, ldu, work, rwork, result.GetPtr(7))

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
				(*llwork)[j-1] = true
			}
			for j = i1 + 1; j <= n; j++ {
				(*llwork)[j-1] = false
			}

			golapack.Ztgevc('L', 'S', *llwork, &n, s1, lda, p1, lda, evectl, ldu, cdumma.CMatrix(*ldu, opts), ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTGEVC(L,S1)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			i1 = in
			for j = 1; j <= i1; j++ {
				(*llwork)[j-1] = false
			}
			for j = i1 + 1; j <= n; j++ {
				(*llwork)[j-1] = true
			}

			golapack.Ztgevc('L', 'S', *llwork, &n, s1, lda, p1, lda, evectl.Off(0, i1+1-1), ldu, cdumma.CMatrix(*ldu, opts), ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTGEVC(L,S2)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			Zget52(true, &n, s1, lda, p1, lda, evectl, ldu, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(8, dumma.Get(0))
			if dumma.Get(1) > (*thrshn) {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "ZTGEVC(HOWMNY=S)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           10: Compute the left eigenvector Matrix with
			//               back transforming:
			ntest = 10
			result.Set(9, ulpinv)
			golapack.Zlacpy('F', &n, &n, q, ldu, evectl, ldu)
			golapack.Ztgevc('L', 'B', *llwork, &n, s1, lda, p1, lda, evectl, ldu, cdumma.CMatrix(*ldu, opts), ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTGEVC(L,B)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			Zget52(true, &n, h, lda, t, lda, evectl, ldu, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(9, dumma.Get(0))
			if dumma.Get(1) > (*thrshn) {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "ZTGEVC(HOWMNY=B)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           11: Compute the right eigenvector Matrix without
			//               back transforming:
			ntest = 11
			result.Set(10, ulpinv)

			//           To test "SELECT" option, compute half of the eigenvectors
			//           in one call, and half in another
			i1 = n / 2
			for j = 1; j <= i1; j++ {
				(*llwork)[j-1] = true
			}
			for j = i1 + 1; j <= n; j++ {
				(*llwork)[j-1] = false
			}

			golapack.Ztgevc('R', 'S', *llwork, &n, s1, lda, p1, lda, cdumma.CMatrix(*ldu, opts), ldu, evectr, ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTGEVC(R,S1)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			i1 = in
			for j = 1; j <= i1; j++ {
				(*llwork)[j-1] = false
			}
			for j = i1 + 1; j <= n; j++ {
				(*llwork)[j-1] = true
			}

			golapack.Ztgevc('R', 'S', *llwork, &n, s1, lda, p1, lda, cdumma.CMatrix(*ldu, opts), ldu, evectr.Off(0, i1+1-1), ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTGEVC(R,S2)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			Zget52(false, &n, s1, lda, p1, lda, evectr, ldu, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(10, dumma.Get(0))
			if dumma.Get(1) > (*thresh) {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "ZTGEVC(HOWMNY=S)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           12: Compute the right eigenvector Matrix with
			//               back transforming:
			ntest = 12
			result.Set(11, ulpinv)
			golapack.Zlacpy('F', &n, &n, z, ldu, evectr, ldu)
			golapack.Ztgevc('R', 'B', *llwork, &n, s1, lda, p1, lda, cdumma.CMatrix(*ldu, opts), ldu, evectr, ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTGEVC(R,B)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label210
			}

			Zget52(false, &n, h, lda, t, lda, evectr, ldu, alpha1, beta1, work, rwork, dumma.Off(0))
			result.Set(11, dumma.Get(0))
			if dumma.Get(1) > (*thresh) {
				_t.Fail()
				fmt.Printf(" ZCHKGG: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "ZTGEVC(HOWMNY=B)", dumma.Get(1), n, jtype, ioldsd)
			}

			//           Tests 13--15 are done only on request
			if *tstdif {
				//              Do Tests 13--14
				Zget51(func() *int { y := 2; return &y }(), &n, s1, lda, s2, lda, q, ldu, z, ldu, work, rwork, result.GetPtr(12))
				Zget51(func() *int { y := 2; return &y }(), &n, p1, lda, p2, lda, q, ldu, z, ldu, work, rwork, result.GetPtr(13))

				//              Do Test 15
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, cmplx.Abs(alpha1.Get(j-1)-alpha3.Get(j-1)))
					temp2 = maxf64(temp2, cmplx.Abs(beta1.Get(j-1)-beta3.Get(j-1)))
				}

				temp1 = temp1 / maxf64(safmin, ulp*maxf64(temp1, anorm))
				temp2 = temp2 / maxf64(safmin, ulp*maxf64(temp2, bnorm))
				result.Set(14, maxf64(temp1, temp2))
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
				if result.Get(jr-1) >= (*thresh) {
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf(" %3s -- Complex Generalized eigenvalue problem\n", "ZGG")

						//                    Matrix types
						fmt.Printf(" Matrix types (see ZCHKGG for details): \n")
						fmt.Printf(" Special Matrices:                       (J'=transposed Jordan block)\n   1=(0,0)  2=(I,0)  3=(0,I)  4=(I,I)  5=(J',J')  6=(diag(J',I), diag(I,J'))\n Diagonal Matrices:  ( D=diag(0,1,2,...) )\n   7=(D,I)   9=(large*D, small*I)  11=(large*I, small*D)  13=(large*D, large*I)\n   8=(I,D)  10=(small*D, large*I)  12=(small*I, large*D)  14=(small*D, small*I)\n  15=(D, reversed D)\n")
						fmt.Printf(" Matrices Rotated by Random %s Matrices U, V:\n  16=Transposed Jordan Blocks             19=geometric alpha, beta=0,1\n  17=arithm. alpha&beta                   20=arithmetic alpha, beta=0,1\n  18=clustered alpha, beta=0,1            21=random alpha, beta=0,1\n Large & Small Matrices:\n  22=(large, small)   23=(small,large)    24=(small,small)    25=(large,large)\n  26=random O(1) matrices.\n", "Unitary")

						//                    Tests performed
						fmt.Printf("\n Tests performed:   (H is Hessenberg, S is Schur, B, T, P are triangular,\n                    U, V, Q, and Z are %s, l and r are the\n                    appropriate left and right eigenvectors, resp., a is\n                    alpha, b is beta, and %s means %s.)\n 1 = | A - U H V%s | / ( |A| n ulp )      2 = | B - U T V%s | / ( |B| n ulp )\n 3 = | I - UU%s | / ( n ulp )             4 = | I - VV%s | / ( n ulp )\n 5 = | H - Q S Z%s | / ( |H| n ulp )      6 = | T - Q P Z%s | / ( |T| n ulp )\n 7 = | I - QQ%s | / ( n ulp )             8 = | I - ZZ%s | / ( n ulp )\n 9 = maxint | ( b S - a P )%s l | / const.  10 = maxint | ( b H - a T )%s l | / const.\n 11= maxint | ( b S - a P ) r | / const.   12 = maxint | ( b H - a T ) r | / const.\n \n", "unitary", "*", "conjugate transpose", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*")

					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}

		label230:
		}
	}

	//     Summary
	Dlasum([]byte("ZGG"), &nerrs, &ntestt)
}
