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

// Ddrgev checks the nonsymmetric generalized eigenvalue problem driver
// routine DGGEV.
//
// DGGEV computes for a pair of n-by-n nonsymmetric matrices (A,B) the
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
// When DDRGEV is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each type of matrix, a pair of matrices (A, B) will be generated
// and used for testing.  For each matrix pair, the following tests
// will be performed and compared with the threshold THRESH.
//
// Results from DGGEV:
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
func Ddrgev(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.Matrix, lda *int, b, s, t, q *mat.Matrix, ldq *int, z, qe *mat.Matrix, ldqe *int, alphar, alphai, beta, alphr1, alphi1, beta1, work *mat.Vector, lwork *int, result *mat.Vector, info *int, _t *testing.T) {
	var badnn bool
	var one, safmax, safmin, ulp, ulpinv, zero float64
	var i, iadd, ierr, in, j, jc, jr, jsize, jtype, maxtyp, maxwrk, minwrk, mtypes, n, n1, nerrs, nmats, nmax, ntestt int

	rmagn := vf(4)
	iasign := make([]int, 26)
	ibsign := make([]int, 26)
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
	iasign[0], iasign[1], iasign[2], iasign[3], iasign[4], iasign[5], iasign[6], iasign[7], iasign[8], iasign[9], iasign[10], iasign[11], iasign[12], iasign[13], iasign[14], iasign[15], iasign[16], iasign[17], iasign[18], iasign[19], iasign[20], iasign[21], iasign[22], iasign[23], iasign[24], iasign[25] = 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0
	ibsign[0], ibsign[1], ibsign[2], ibsign[3], ibsign[4], ibsign[5], ibsign[6], ibsign[7], ibsign[8], ibsign[9], ibsign[10], ibsign[11], ibsign[12], ibsign[13], ibsign[14], ibsign[15], ibsign[16], ibsign[17], ibsign[18], ibsign[19], ibsign[20], ibsign[21], ibsign[22], ibsign[23], ibsign[24], ibsign[25] = 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0

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
	} else if (*ldqe) <= 1 || (*ldqe) < nmax {
		(*info) = -17
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	minwrk = 1
	if (*info) == 0 && (*lwork) >= 1 {
		minwrk = maxint(1, 8*nmax, nmax*(nmax+1))
		maxwrk = 7*nmax + nmax*Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQRF"), []byte{' '}, &nmax, func() *int { y := 1; return &y }(), &nmax, func() *int { y := 0; return &y }())
		maxwrk = maxint(maxwrk, nmax*(nmax+1))
		work.Set(0, float64(maxwrk))
	}

	if (*lwork) < minwrk {
		(*info) = -25
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DDRGEV"), -(*info))
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
				goto label210
			}
			nmats = nmats + 1

			//           Save ISEED in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
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
				goto label100
			}
			ierr = 0
			if kclass[jtype-1] < 3 {
				//              Generate A (w/o rotation)
				if absint(katype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Dlaset('F', &n, &n, &zero, &zero, a, lda)
					}
				} else {
					in = n
				}
				Dlatm4(&(katype[jtype-1]), &in, &(kz1[kazero[jtype-1]-1]), &(kz2[kazero[jtype-1]-1]), &(iasign[jtype-1]), rmagn.GetPtr(kamagn[jtype-1]-0), &ulp, rmagn.GetPtr(ktrian[jtype-1]*kamagn[jtype-1]-0), func() *int { y := 2; return &y }(), iseed, a, lda)
				iadd = kadd[kazero[jtype-1]-1]
				if iadd > 0 && iadd <= n {
					a.Set(iadd-1, iadd-1, one)
				}

				//              Generate B (w/o rotation)
				if absint(kbtype[jtype-1]) == 3 {
					in = 2*((n-1)/2) + 1
					if in != n {
						golapack.Dlaset('F', &n, &n, &zero, &zero, b, lda)
					}
				} else {
					in = n
				}
				Dlatm4(&(kbtype[jtype-1]), &in, &(kz1[kbzero[jtype-1]-1]), &(kz2[kbzero[jtype-1]-1]), &(ibsign[jtype-1]), rmagn.GetPtr(kbmagn[jtype-1]-0), &one, rmagn.GetPtr(ktrian[jtype-1]*kbmagn[jtype-1]-0), func() *int { y := 2; return &y }(), iseed, b, lda)
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
							q.Set(jr-1, jc-1, matgen.Dlarnd(func() *int { y := 3; return &y }(), iseed))
							z.Set(jr-1, jc-1, matgen.Dlarnd(func() *int { y := 3; return &y }(), iseed))
						}
						golapack.Dlarfg(toPtr(n+1-jc), q.GetPtr(jc-1, jc-1), q.Vector(jc+1-1, jc-1), func() *int { y := 1; return &y }(), work.GetPtr(jc-1))
						work.Set(2*n+jc-1, math.Copysign(one, q.Get(jc-1, jc-1)))
						q.Set(jc-1, jc-1, one)
						golapack.Dlarfg(toPtr(n+1-jc), z.GetPtr(jc-1, jc-1), z.Vector(jc+1-1, jc-1), func() *int { y := 1; return &y }(), work.GetPtr(n+jc-1))
						work.Set(3*n+jc-1, math.Copysign(one, z.Get(jc-1, jc-1)))
						z.Set(jc-1, jc-1, one)
					}
					q.Set(n-1, n-1, one)
					work.Set(n-1, zero)
					work.Set(3*n-1, math.Copysign(one, matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed)))
					z.Set(n-1, n-1, one)
					work.Set(2*n-1, zero)
					work.Set(4*n-1, math.Copysign(one, matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed)))

					//                 Apply the diagonal matrices
					for jc = 1; jc <= n; jc++ {
						for jr = 1; jr <= n; jr++ {
							a.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.Get(3*n+jc-1)*a.Get(jr-1, jc-1))
							b.Set(jr-1, jc-1, work.Get(2*n+jr-1)*work.Get(3*n+jc-1)*b.Get(jr-1, jc-1))
						}
					}
					golapack.Dorm2r('L', 'N', &n, &n, toPtr(n-1), q, ldq, work, a, lda, work.Off(2*n+1-1), &ierr)
					if ierr != 0 {
						goto label90
					}
					golapack.Dorm2r('R', 'T', &n, &n, toPtr(n-1), z, ldq, work.Off(n+1-1), a, lda, work.Off(2*n+1-1), &ierr)
					if ierr != 0 {
						goto label90
					}
					golapack.Dorm2r('L', 'N', &n, &n, toPtr(n-1), q, ldq, work, b, lda, work.Off(2*n+1-1), &ierr)
					if ierr != 0 {
						goto label90
					}
					golapack.Dorm2r('R', 'T', &n, &n, toPtr(n-1), z, ldq, work.Off(n+1-1), b, lda, work.Off(2*n+1-1), &ierr)
					if ierr != 0 {
						goto label90
					}
				}
			} else {
				//              Random matrices
				for jc = 1; jc <= n; jc++ {
					for jr = 1; jr <= n; jr++ {
						a.Set(jr-1, jc-1, rmagn.Get(kamagn[jtype-1]-0)*matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed))
						b.Set(jr-1, jc-1, rmagn.Get(kbmagn[jtype-1]-0)*matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed))
					}
				}
			}

		label90:
			;

			if ierr != 0 {
				_t.Fail()
				fmt.Printf(" DDRGEV: %s returned INFO=%6d.\n   N=%6d, JTYPE=%6d, ISEED=%4d\n", "Generator", ierr, n, jtype, ioldsd)
				(*info) = absint(ierr)
				return
			}

		label100:
			;

			for i = 1; i <= 7; i++ {
				result.Set(i-1, -one)
			}

			//           Call DGGEV to compute eigenvalues and eigenvectors.
			golapack.Dlacpy(' ', &n, &n, a, lda, s, lda)
			golapack.Dlacpy(' ', &n, &n, b, lda, t, lda)
			golapack.Dggev('V', 'V', &n, s, lda, t, lda, alphar, alphai, beta, q, ldq, z, ldq, work, lwork, &ierr)
			if ierr != 0 && ierr != n+1 {
				_t.Fail()
				result.Set(0, ulpinv)
				fmt.Printf(" DDRGEV: %s returned INFO=%6d.\n   N=%6d, JTYPE=%6d, ISEED=%4d\n", "DGGEV1", ierr, n, jtype, ioldsd)
				(*info) = absint(ierr)
				goto label190
			}

			//           Do the tests (1) and (2)
			Dget52(true, &n, a, lda, b, lda, q, ldq, alphar, alphai, beta, work, result.Off(0))
			if result.Get(1) > (*thresh) {
				_t.Fail()
				fmt.Printf(" DDRGEV: %s Eigenvectors from %s incorrectly normalized.\n Bits of error= %10.3f,   N=%4d, JTYPE=%3d, ISEED=%4d\n", "Left", "DGGEV1", result.Get(1), n, jtype, ioldsd)
			}

			//           Do the tests (3) and (4)
			Dget52(false, &n, a, lda, b, lda, z, ldq, alphar, alphai, beta, work, result.Off(2))
			if result.Get(3) > (*thresh) {
				_t.Fail()
				fmt.Printf(" DDRGEV: %s Eigenvectors from %s incorrectly normalized.\n Bits of error= %10.3f,   N=%4d, JTYPE=%3d, ISEED=%4d\n", "Right", "DGGEV1", result.Get(3), n, jtype, ioldsd)
			}

			//           Do the test (5)
			golapack.Dlacpy(' ', &n, &n, a, lda, s, lda)
			golapack.Dlacpy(' ', &n, &n, b, lda, t, lda)
			golapack.Dggev('N', 'N', &n, s, lda, t, lda, alphr1, alphi1, beta1, q, ldq, z, ldq, work, lwork, &ierr)
			if ierr != 0 && ierr != n+1 {
				_t.Fail()
				result.Set(0, ulpinv)
				fmt.Printf(" DDRGEV: %s returned INFO=%6d.\n   N=%6d, JTYPE=%6d, ISEED=%4d\n", "DGGEV2", ierr, n, jtype, ioldsd)
				(*info) = absint(ierr)
				goto label190
			}

			for j = 1; j <= n; j++ {
				if alphar.Get(j-1) != alphr1.Get(j-1) || alphai.Get(j-1) != alphi1.Get(j-1) || beta.Get(j-1) != beta1.Get(j-1) {
					result.Set(4, ulpinv)
				}
			}

			//           Do the test (6): Compute eigenvalues and left eigenvectors,
			//           and test them
			golapack.Dlacpy(' ', &n, &n, a, lda, s, lda)
			golapack.Dlacpy(' ', &n, &n, b, lda, t, lda)
			golapack.Dggev('V', 'N', &n, s, lda, t, lda, alphr1, alphi1, beta1, qe, ldqe, z, ldq, work, lwork, &ierr)
			if ierr != 0 && ierr != n+1 {
				_t.Fail()
				result.Set(0, ulpinv)
				fmt.Printf(" DDRGEV: %s returned INFO=%6d.\n   N=%6d, JTYPE=%6d, ISEED=%4d\n", "DGGEV3", ierr, n, jtype, ioldsd)
				(*info) = absint(ierr)
				goto label190
			}

			for j = 1; j <= n; j++ {
				if alphar.Get(j-1) != alphr1.Get(j-1) || alphai.Get(j-1) != alphi1.Get(j-1) || beta.Get(j-1) != beta1.Get(j-1) {
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

			//           DO the test (7): Compute eigenvalues and right eigenvectors,
			//           and test them
			golapack.Dlacpy(' ', &n, &n, a, lda, s, lda)
			golapack.Dlacpy(' ', &n, &n, b, lda, t, lda)
			golapack.Dggev('N', 'V', &n, s, lda, t, lda, alphr1, alphi1, beta1, q, ldq, qe, ldqe, work, lwork, &ierr)
			if ierr != 0 && ierr != n+1 {
				_t.Fail()
				result.Set(0, ulpinv)
				fmt.Printf(" DDRGEV: %s returned INFO=%6d.\n   N=%6d, JTYPE=%6d, ISEED=%4d\n", "DGGEV4", ierr, n, jtype, ioldsd)
				(*info) = absint(ierr)
				goto label190
			}

			for j = 1; j <= n; j++ {
				if alphar.Get(j-1) != alphr1.Get(j-1) || alphai.Get(j-1) != alphi1.Get(j-1) || beta.Get(j-1) != beta1.Get(j-1) {
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
				if result.Get(jr-1) >= (*thresh) {
					_t.Fail()
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Real Generalized eigenvalue problem driver\n", "DGV")

						//                    Matrix types
						fmt.Printf(" Matrix types (see DDRGEV for details): \n")
						fmt.Printf(" Special Matrices:                       (J'=transposed Jordan block)\n   1=(0,0)  2=(I,0)  3=(0,I)  4=(I,I)  5=(J',J')  6=(diag(J',I), diag(I,J'))\n Diagonal Matrices:  ( D=diag(0,1,2,...) )\n   7=(D,I)   9=(large*D, small*I)  11=(large*I, small*D)  13=(large*D, large*I)\n   8=(I,D)  10=(small*D, large*I)  12=(small*I, large*D)  14=(small*D, small*I)\n  15=(D, reversed D)\n")
						fmt.Printf(" Matrices Rotated by Random %s Matrices U, V:\n  16=Transposed Jordan Blocks             19=geometric alpha, beta=0,1\n  17=arithm. alpha&beta                   20=arithmetic alpha, beta=0,1\n  18=clustered alpha, beta=0,1            21=random alpha, beta=0,1\n Large & Small Matrices:\n  22=(large, small)   23=(small,large)    24=(small,small)    25=(large,large)\n  26=random O(1) matrices.\n", "Orthogonal")

						//                    Tests performed
						fmt.Printf("\n Tests performed:    \n 1 = max | ( b A - a B )'*l | / const.,\n 2 = | |VR(i)| - 1 | / ulp,\n 3 = max | ( b A - a B )*r | / const.\n 4 = | |VL(i)| - 1 | / ulp,\n 5 = 0 if W same no matter if r or l computed,\n 6 = 0 if l same no matter if l computed,\n 7 = 0 if r same no matter if r computed,\n \n")

					}
					nerrs = nerrs + 1
					if result.Get(jr-1) < 10000.0 {
						fmt.Printf(" Matrix order=%5d, type=%2d, seed=%4d, result %2d is %8.2f\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					} else {
						fmt.Printf(" Matrix order=%5d, type=%2d, seed=%4d, result %2d is %10.3E\n", n, jtype, ioldsd, jr, result.Get(jr-1))
					}
				}
			}

		label210:
		}
	}

	//     Summary
	Alasvm([]byte("DGV"), &nerrs, &ntestt, func() *int { y := 0; return &y }())

	work.Set(0, float64(maxwrk))
}
