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

// Zdrges checks the nonsymmetric generalized eigenvalue (Schur form)
// problem driver ZGGES.
//
// ZGGES factors A and B as Q*S*Z'  and Q*T*Z' , where ' means conjugate
// transpose, S and T are  upper triangular (i.e., in generalized Schur
// form), and Q and Z are unitary. It also computes the generalized
// eigenvalues (alpha(j),beta(j)), j=1,...,n.  Thus,
// w(j) = alpha(j)/beta(j) is a root of the characteristic equation
//
//                 det( A - w(j) B ) = 0
//
// Optionally it also reorder the eigenvalues so that a selected
// cluster of eigenvalues appears in the leading diagonal block of the
// Schur forms.
//
// When ZDRGES is called, a number of matrix "sizes" ("N's") and a
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
// (5)   if A is in Schur form (i.e. triangular form) (no sorting of
//       eigenvalues)
//
// (6)   if eigenvalues = diagonal elements of the Schur form (S, T),
//       i.e., test the maximum over j of D(j)  where:
//
//                     |alpha(j) - S(j,j)|        |beta(j) - T(j,j)|
//           D(j) = ------------------------ + -----------------------
//                  max(|alpha(j)|,|S(j,j)|)   max(|beta(j)|,|T(j,j)|)
//
//       (no sorting of eigenvalues)
//
// (7)   | (A,B) - Q (S,T) Z' | / ( |(A,B)| n ulp )
//       (with sorting of eigenvalues).
//
// (8)   | I - QQ' | / ( n ulp ) (with sorting of eigenvalues).
//
// (9)   | I - ZZ' | / ( n ulp ) (with sorting of eigenvalues).
//
// (10)  if A is in Schur form (i.e. quasi-triangular form)
//       (with sorting of eigenvalues).
//
// (11)  if eigenvalues = diagonal elements of the Schur form (S, T),
//       i.e. test the maximum over j of D(j)  where:
//
//                     |alpha(j) - S(j,j)|        |beta(j) - T(j,j)|
//           D(j) = ------------------------ + -----------------------
//                  max(|alpha(j)|,|S(j,j)|)   max(|beta(j)|,|T(j,j)|)
//
//       (with sorting of eigenvalues).
//
// (12)  if sorting worked and SDIM is the number of eigenvalues
//       which were CELECTed.
//
// Test Matrices
// =============
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
func Zdrges(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.CMatrix, lda *int, b, s, t, q *mat.CMatrix, ldq *int, z *mat.CMatrix, alpha, beta, work *mat.CVector, lwork *int, rwork, result *mat.Vector, bwork *[]bool, info *int, _t *testing.T) {
	var badnn, ilabad bool
	var sort byte
	var cone, ctemp, czero complex128
	var one, safmax, safmin, temp1, temp2, ulp, ulpinv, zero float64
	var i, iadd, iinfo, in, isort, j, jc, jr, jsize, jtype, knteig, maxtyp, maxwrk, minwrk, mtypes, n, n1, nb, nerrs, nmats, nmax, ntest, ntestt, rsub, sdim int
	lasign := make([]bool, 26)
	lbsign := make([]bool, 26)
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
		nmax = max(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
	}

	if (*nsizes) < 0 {
		(*info) = -1
	} else if badnn {
		(*info) = -2
	} else if (*ntypes) < 0 {
		(*info) = -3
	} else if (*thresh) < zero {
		(*info) = -6
	} else if (*lda) <= 1 || (*lda) < nmax {
		(*info) = -9
	} else if (*ldq) <= 1 || (*ldq) < nmax {
		(*info) = -14
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	minwrk = 1
	if (*info) == 0 && (*lwork) >= 1 {
		minwrk = 3 * nmax * nmax
		nb = max(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, &nmax, &nmax, toPtr(-1), toPtr(-1)), Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LC"), &nmax, &nmax, &nmax, toPtr(-1)), Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQR"), []byte{' '}, &nmax, &nmax, &nmax, toPtr(-1)))
		maxwrk = max(nmax+nmax*nb, 3*nmax*nmax)
		work.SetRe(0, float64(maxwrk))
	}

	if (*lwork) < minwrk {
		(*info) = -19
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZDRGES"), -(*info))
		return
	}

	//     Quick return if possible
	if (*nsizes) == 0 || (*ntypes) == 0 {
		return
	}

	ulp = golapack.Dlamch(Precision)
	safmin = golapack.Dlamch(SafeMinimum)
	safmin = safmin / ulp
	safmax = one / safmin
	golapack.Dlabad(&safmin, &safmax)
	ulpinv = one / ulp

	//     The values RMAGN(2:3) depend on N, see below.
	rmagn.Set(0, zero)
	rmagn.Set(1, one)

	//     Loop over matrix sizes
	ntestt = 0
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		n = (*nn)[jsize-1]
		n1 = max(1, n)
		rmagn.Set(2, safmax*ulp/float64(n1))
		rmagn.Set(3, safmin*ulpinv*float64(n1))

		if (*nsizes) != 1 {
			mtypes = min(maxtyp, *ntypes)
		} else {
			mtypes = min(maxtyp+1, *ntypes)
		}

		//        Loop over matrix types
		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label180
			}
			nmats = nmats + 1
			ntest = 0

			//           Save ISEED in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
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
				goto label110
			}
			iinfo = 0
			if kclass[jtype-1] < 3 {
				//              Generate A (w/o rotation)
				if abs(katype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Zlaset('F', &n, &n, &czero, &czero, a, lda)
					}
				} else {
					in = n
				}
				Zlatm4(&katype[jtype-1], &in, &kz1[kazero[jtype-1]-1], &kz2[kazero[jtype-1]-1], lasign[jtype-1], rmagn.GetPtr(kamagn[jtype-1]-0), &ulp, rmagn.GetPtr(ktrian[jtype-1]*kamagn[jtype-1]-0), func() *int { y := 2; return &y }(), iseed, a, lda)
				iadd = kadd[kazero[jtype-1]-1]
				if iadd > 0 && iadd <= n {
					a.SetRe(iadd-1, iadd-1, rmagn.Get(kamagn[jtype-1]-0))
				}

				//              Generate B (w/o rotation)
				if abs(kbtype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Zlaset('F', &n, &n, &czero, &czero, b, lda)
					}
				} else {
					in = n
				}
				Zlatm4(&kbtype[jtype-1], &in, &kz1[kbzero[jtype-1]-1], &kz2[kbzero[jtype-1]-1], lbsign[jtype-1], rmagn.GetPtr(kbmagn[jtype-1]-0), &one, rmagn.GetPtr(ktrian[jtype-1]*kbmagn[jtype-1]-0), func() *int { y := 2; return &y }(), iseed, b, lda)
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
							q.Set(jr-1, jc-1, matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed))
							z.Set(jr-1, jc-1, matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed))
						}
						golapack.Zlarfg(toPtr(n+1-jc), q.GetPtr(jc-1, jc-1), q.CVector(jc, jc-1), func() *int { y := 1; return &y }(), work.GetPtr(jc-1))
						work.SetRe(2*n+jc-1, math.Copysign(one, q.GetRe(jc-1, jc-1)))
						q.Set(jc-1, jc-1, cone)
						golapack.Zlarfg(toPtr(n+1-jc), z.GetPtr(jc-1, jc-1), z.CVector(jc, jc-1), func() *int { y := 1; return &y }(), work.GetPtr(n+jc-1))
						work.SetRe(3*n+jc-1, math.Copysign(one, z.GetRe(jc-1, jc-1)))
						z.Set(jc-1, jc-1, cone)
					}
					ctemp = matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed)
					q.Set(n-1, n-1, cone)
					work.Set(n-1, czero)
					work.Set(3*n-1, ctemp/complex(cmplx.Abs(ctemp), 0))
					ctemp = matgen.Zlarnd(func() *int { y := 3; return &y }(), iseed)
					z.Set(n-1, n-1, cone)
					work.Set(2*n-1, czero)
					work.Set(4*n-1, ctemp/complex(cmplx.Abs(ctemp), 0))

					//                 Apply the diagonal matrices
					for jc = 1; jc <= n; jc++ {
						for jr = 1; jr <= n; jr++ {
							a.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.GetConj(3*n+jc-1)*a.Get(jr-1, jc-1))
							b.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.GetConj(3*n+jc-1)*b.Get(jr-1, jc-1))
						}
					}
					golapack.Zunm2r('L', 'N', &n, &n, toPtr(n-1), q, ldq, work, a, lda, work.Off(2*n), &iinfo)
					if iinfo != 0 {
						goto label100
					}
					golapack.Zunm2r('R', 'C', &n, &n, toPtr(n-1), z, ldq, work.Off(n), a, lda, work.Off(2*n), &iinfo)
					if iinfo != 0 {
						goto label100
					}
					golapack.Zunm2r('L', 'N', &n, &n, toPtr(n-1), q, ldq, work, b, lda, work.Off(2*n), &iinfo)
					if iinfo != 0 {
						goto label100
					}
					golapack.Zunm2r('R', 'C', &n, &n, toPtr(n-1), z, ldq, work.Off(n), b, lda, work.Off(2*n), &iinfo)
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

		label100:
			;

			if iinfo != 0 {
				_t.Fail()
				fmt.Printf(" ZDRGES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = abs(iinfo)
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

				//              Call ZGGES to compute H, T, Q, Z, alpha, and beta.
				golapack.Zlacpy('F', &n, &n, a, lda, s, lda)
				golapack.Zlacpy('F', &n, &n, b, lda, t, lda)
				ntest = 1 + rsub + isort
				result.Set(1+rsub+isort-1, ulpinv)
				golapack.Zgges('V', 'V', sort, Zlctes, &n, s, lda, t, lda, &sdim, alpha, beta, q, ldq, z, ldq, work, lwork, rwork, bwork, &iinfo)
				if iinfo != 0 && iinfo != n+2 {
					_t.Fail()
					result.Set(1+rsub+isort-1, ulpinv)
					fmt.Printf(" ZDRGES: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGGES", iinfo, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					goto label160
				}

				ntest = 4 + rsub

				//              Do tests 1--4 (or tests 7--9 when reordering )
				if isort == 0 {
					Zget51(func() *int { y := 1; return &y }(), &n, a, lda, s, lda, q, ldq, z, ldq, work, rwork, result.GetPtr(0))
					Zget51(func() *int { y := 1; return &y }(), &n, b, lda, t, lda, q, ldq, z, ldq, work, rwork, result.GetPtr(1))
				} else {
					Zget54(&n, a, lda, b, lda, s, lda, t, lda, q, ldq, z, ldq, work, result.GetPtr(2+rsub-1))
				}

				Zget51(func() *int { y := 3; return &y }(), &n, b, lda, t, lda, q, ldq, q, ldq, work, rwork, result.GetPtr(3+rsub-1))
				Zget51(func() *int { y := 3; return &y }(), &n, b, lda, t, lda, z, ldq, z, ldq, work, rwork, result.GetPtr(4+rsub-1))

				//              Do test 5 and 6 (or Tests 10 and 11 when reordering):
				//              check Schur form of A and compare eigenvalues with
				//              diagonals.
				ntest = 6 + rsub
				temp1 = zero

				for j = 1; j <= n; j++ {
					ilabad = false
					temp2 = (abs1(alpha.Get(j-1)-s.Get(j-1, j-1))/math.Max(safmin, math.Max(abs1(alpha.Get(j-1)), abs1(s.Get(j-1, j-1)))) + abs1(beta.Get(j-1)-t.Get(j-1, j-1))/math.Max(safmin, math.Max(abs1(beta.Get(j-1)), abs1(t.Get(j-1, j-1))))) / ulp

					if j < n {
						if s.GetRe(j, j-1) != zero {
							ilabad = true
							result.Set(5+rsub-1, ulpinv)
						}
					}
					if j > 1 {
						if s.GetRe(j-1, j-1-1) != zero {
							ilabad = true
							result.Set(5+rsub-1, ulpinv)
						}
					}
					temp1 = math.Max(temp1, temp2)
					if ilabad {
						fmt.Printf(" ZDRGES: S not in Schur form at eigenvalue %6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", j, n, jtype, ioldsd)
					}
				}
				result.Set(6+rsub-1, temp1)

				if isort >= 1 {
					//                 Do test 12
					ntest = 12
					result.Set(11, zero)
					knteig = 0
					for i = 1; i <= n; i++ {
						if Zlctes(alpha.Get(i-1), beta.Get(i-1)) {
							knteig = knteig + 1
						}
					}
					if sdim != knteig {
						result.Set(12, ulpinv)
					}
				}

			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label160:
			;

			ntestt = ntestt + ntest

			//           Print out tests which fail.
			for jr = 1; jr <= ntest; jr++ {
				if result.Get(jr-1) >= (*thresh) {
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Complex Generalized Schur from problem driver\n", "ZGS")

						//                    Matrix types
						fmt.Printf(" Matrix types (see ZDRGES for details): \n")
						fmt.Printf(" Special Matrices:                       (J'=transposed Jordan block)\n   1=(0,0)  2=(I,0)  3=(0,I)  4=(I,I)  5=(J',J')  6=(diag(J',I), diag(I,J'))\n Diagonal Matrices:  ( D=diag(0,1,2,...) )\n   7=(D,I)   9=(large*D, small*I)  11=(large*I, small*D)  13=(large*D, large*I)\n   8=(I,D)  10=(small*D, large*I)  12=(small*I, large*D)  14=(small*D, small*I)\n  15=(D, reversed D)\n")
						fmt.Printf(" Matrices Rotated by Random %s Matrices U, V:\n  16=Transposed Jordan Blocks             19=geometric alpha, beta=0,1\n  17=arithm. alpha&beta                   20=arithmetic alpha, beta=0,1\n  18=clustered alpha, beta=0,1            21=random alpha, beta=0,1\n Large & Small Matrices:\n  22=(large, small)   23=(small,large)    24=(small,small)    25=(large,large)\n  26=random O(1) matrices.\n", "Unitary")

						//                    Tests performed
						fmt.Printf("\n Tests performed:  (S is Schur, T is triangular, Q and Z are %s,\n                   l and r are the appropriate left and right\n                   eigenvectors, resp., a is alpha, b is beta, and\n                   %s means %s.)\n Without ordering: \n  1 = | A - Q S Z%s | / ( |A| n ulp )      2 = | B - Q T Z%s | / ( |B| n ulp )\n  3 = | I - QQ%s | / ( n ulp )             4 = | I - ZZ%s | / ( n ulp )\n  5 = A is in Schur form S\n  6 = difference between (alpha,beta) and diagonals of (S,T)\n With ordering: \n  7 = | (A,B) - Q (S,T) Z%s | / ( |(A,B)| n ulp )\n  8 = | I - QQ%s | / ( n ulp )             9 = | I - ZZ%s | / ( n ulp )\n 10 = A is in Schur form S\n 11 = difference between (alpha,beta) and diagonals of (S,T)\n 12 = SDIM is the correct number of selected eigenvalues\n\n", "unitary", "'", "transpose", "'", "'", "'", "'", "'", "'", "'")

					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %2d is%10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}

		label180:
		}
	}

	//     Summary
	Alasvm([]byte("ZGS"), &nerrs, &ntestt, func() *int { y := 0; return &y }())

	work.SetRe(0, float64(maxwrk))
}
