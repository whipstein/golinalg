package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggsvd3 computes the generalized singular value decomposition (GSVD)
// of an M-by-N real matrix A and P-by-N real matrix B:
//
//       U**T*A*Q = D1*( 0 R ),    V**T*B*Q = D2*( 0 R )
//
// where U, V and Q are orthogonal matrices.
// Let K+L = the effective numerical rank of the matrix (A**T,B**T)**T,
// then R is a K+L-by-K+L nonsingular upper triangular matrix, D1 and
// D2 are M-by-(K+L) and P-by-(K+L) "diagonal" matrices and of the
// following structures, respectively:
//
// If M-K-L >= 0,
//
//                     K  L
//        D1 =     K ( I  0 )
//                 L ( 0  C )
//             M-K-L ( 0  0 )
//
//                   K  L
//        D2 =   L ( 0  S )
//             P-L ( 0  0 )
//
//                 N-K-L  K    L
//   ( 0 R ) = K (  0   R11  R12 )
//             L (  0    0   R22 )
//
// where
//
//   C = diag( ALPHA(K+1), ... , ALPHA(K+L) ),
//   S = diag( BETA(K+1),  ... , BETA(K+L) ),
//   C**2 + S**2 = I.
//
//   R is stored in A(1:K+L,N-K-L+1:N) on exit.
//
// If M-K-L < 0,
//
//                   K M-K K+L-M
//        D1 =   K ( I  0    0   )
//             M-K ( 0  C    0   )
//
//                     K M-K K+L-M
//        D2 =   M-K ( 0  S    0  )
//             K+L-M ( 0  0    I  )
//               P-L ( 0  0    0  )
//
//                    N-K-L  K   M-K  K+L-M
//   ( 0 R ) =     K ( 0    R11  R12  R13  )
//               M-K ( 0     0   R22  R23  )
//             K+L-M ( 0     0    0   R33  )
//
// where
//
//   C = diag( ALPHA(K+1), ... , ALPHA(M) ),
//   S = diag( BETA(K+1),  ... , BETA(M) ),
//   C**2 + S**2 = I.
//
//   (R11 R12 R13 ) is stored in A(1:M, N-K-L+1:N), and R33 is stored
//   ( 0  R22 R23 )
//   in B(M-K+1:L,N+M-K-L+1:N) on exit.
//
// The routine computes C, S, R, and optionally the orthogonal
// transformation matrices U, V and Q.
//
// In particular, if B is an N-by-N nonsingular matrix, then the GSVD of
// A and B implicitly gives the SVD of A*inv(B):
//                      A*inv(B) = U*(D1*inv(D2))*V**T.
// If ( A**T,B**T)**T  has orthonormal columns, then the GSVD of A and B is
// also equal to the CS decomposition of A and B. Furthermore, the GSVD
// can be used to derive the solution of the eigenvalue problem:
//                      A**T*A x = lambda* B**T*B x.
// In some literature, the GSVD of A and B is presented in the form
//                  U**T*A*X = ( 0 D1 ),   V**T*B*X = ( 0 D2 )
// where U and V are orthogonal and X is nonsingular, D1 and D2 are
// ``diagonal''.  The former GSVD form can be converted to the latter
// form by taking the nonsingular matrix X as
//
//                      X = Q*( I   0    )
//                            ( 0 inv(R) ).
func Dggsvd3(jobu, jobv, jobq byte, m, n, p, k, l *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, alpha, beta *mat.Vector, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, q *mat.Matrix, ldq *int, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var lquery, wantq, wantu, wantv bool
	var anorm, bnorm, smax, temp, tola, tolb, ulp, unfl float64
	var i, ibnd, isub, j, lwkopt, ncycle int

	//     Decode and test the input parameters
	wantu = jobu == 'U'
	wantv = jobv == 'V'
	wantq = jobq == 'Q'
	lquery = ((*lwork) == -1)
	lwkopt = 1

	//     Test the input arguments
	(*info) = 0
	if !(wantu || jobu == 'N') {
		(*info) = -1
	} else if !(wantv || jobv == 'N') {
		(*info) = -2
	} else if !(wantq || jobq == 'N') {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*p) < 0 {
		(*info) = -6
	} else if (*lda) < maxint(1, *m) {
		(*info) = -10
	} else if (*ldb) < maxint(1, *p) {
		(*info) = -12
	} else if (*ldu) < 1 || (wantu && (*ldu) < (*m)) {
		(*info) = -16
	} else if (*ldv) < 1 || (wantv && (*ldv) < (*p)) {
		(*info) = -18
	} else if (*ldq) < 1 || (wantq && (*ldq) < (*n)) {
		(*info) = -20
	} else if (*lwork) < 1 && !lquery {
		(*info) = -24
	}

	//     Compute workspace
	if (*info) == 0 {
		Dggsvp3(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, &tola, &tolb, k, l, u, ldu, v, ldv, q, ldq, iwork, work, work, toPtr(-1), info)
		lwkopt = (*n) + int(work.Get(0))
		lwkopt = maxint(2*(*n), lwkopt)
		lwkopt = maxint(1, lwkopt)
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGSVD3"), -(*info))
		return
	}
	if lquery {
		return
	}

	//     Compute the Frobenius norm of matrices A and B
	anorm = Dlange('1', m, n, a, lda, work)
	bnorm = Dlange('1', p, n, b, ldb, work)

	//     Get machine precision and set up threshold for determining
	//     the effective numerical rank of the matrices A and B.
	ulp = Dlamch(Precision)
	unfl = Dlamch(SafeMinimum)
	tola = float64(maxint(*m, *n)) * maxf64(anorm, unfl) * ulp
	tolb = float64(maxint(*p, *n)) * maxf64(bnorm, unfl) * ulp

	//     Preprocessing
	Dggsvp3(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, &tola, &tolb, k, l, u, ldu, v, ldv, q, ldq, iwork, work, work.Off((*n)+1-1), toPtr((*lwork)-(*n)), info)

	//     Compute the GSVD of two upper "triangular" matrices
	Dtgsja(jobu, jobv, jobq, m, p, n, k, l, a, lda, b, ldb, &tola, &tolb, alpha, beta, u, ldu, v, ldv, q, ldq, work, &ncycle, info)

	//     Sort the singular values and store the pivot indices in IWORK
	//     Copy ALPHA to WORK, then sort ALPHA in WORK
	goblas.Dcopy(n, alpha, func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
	ibnd = minint(*l, (*m)-(*k))
	for i = 1; i <= ibnd; i++ {
		//        Scan for largest ALPHA(K+I)
		isub = i
		smax = work.Get((*k) + i - 1)
		for j = i + 1; j <= ibnd; j++ {
			temp = work.Get((*k) + j - 1)
			if temp > smax {
				isub = j
				smax = temp
			}
		}
		if isub != i {
			work.Set((*k)+isub-1, work.Get((*k)+i-1))
			work.Set((*k)+i-1, smax)
			(*iwork)[(*k)+i-1] = (*k) + isub
		} else {
			(*iwork)[(*k)+i-1] = (*k) + i
		}
	}

	work.Set(0, float64(lwkopt))
}
