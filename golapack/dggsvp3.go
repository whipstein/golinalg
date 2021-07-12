package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggsvp3 computes orthogonal matrices U, V and Q such that
//
//                    N-K-L  K    L
//  U**T*A*Q =     K ( 0    A12  A13 )  if M-K-L >= 0;
//                 L ( 0     0   A23 )
//             M-K-L ( 0     0    0  )
//
//                  N-K-L  K    L
//         =     K ( 0    A12  A13 )  if M-K-L < 0;
//             M-K ( 0     0   A23 )
//
//                  N-K-L  K    L
//  V**T*B*Q =   L ( 0     0   B13 )
//             P-L ( 0     0    0  )
//
// where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular
// upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0,
// otherwise A23 is (M-K)-by-L upper trapezoidal.  K+L = the effective
// numerical rank of the (M+P)-by-N matrix (A**T,B**T)**T.
//
// This decomposition is the preprocessing step for computing the
// Generalized Singular Value Decomposition (GSVD), see subroutine
// DGGSVD3.
func Dggsvp3(jobu, jobv, jobq byte, m, p, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, tola, tolb *float64, k, l *int, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, q *mat.Matrix, ldq *int, iwork *[]int, tau, work *mat.Vector, lwork, info *int) {
	var forwrd, lquery, wantq, wantu, wantv bool
	var one, zero float64
	var i, j, lwkopt int

	zero = 0.0
	one = 1.0

	//     Test the input parameters
	wantu = jobu == 'U'
	wantv = jobv == 'V'
	wantq = jobq == 'Q'
	forwrd = true
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
	} else if (*p) < 0 {
		(*info) = -5
	} else if (*n) < 0 {
		(*info) = -6
	} else if (*lda) < max(1, *m) {
		(*info) = -8
	} else if (*ldb) < max(1, *p) {
		(*info) = -10
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
		Dgeqp3(p, n, b, ldb, iwork, tau, work, toPtr(-1), info)
		lwkopt = int(work.Get(0))
		if wantv {
			lwkopt = max(lwkopt, *p)
		}
		lwkopt = max(lwkopt, min(*n, *p))
		lwkopt = max(lwkopt, *m)
		if wantq {
			lwkopt = max(lwkopt, *n)
		}
		Dgeqp3(m, n, a, lda, iwork, tau, work, toPtr(-1), info)
		lwkopt = max(lwkopt, int(work.Get(0)))
		lwkopt = max(1, lwkopt)
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGSVP3"), -(*info))
		return
	}
	if lquery {
		return
	}

	//     QR with column pivoting of B: B*P = V*( S11 S12 )
	//                                           (  0   0  )
	for i = 1; i <= (*n); i++ {
		(*iwork)[i-1] = 0
	}
	Dgeqp3(p, n, b, ldb, iwork, tau, work, lwork, info)

	//     Update A := A*P
	Dlapmt(forwrd, m, n, a, lda, iwork)

	//     Determine the effective rank of matrix B.
	(*l) = 0
	for i = 1; i <= min(*p, *n); i++ {
		if math.Abs(b.Get(i-1, i-1)) > (*tolb) {
			(*l) = (*l) + 1
		}
	}

	if wantv {
		//        Copy the details of V, and form V.
		Dlaset('F', p, p, &zero, &zero, v, ldv)
		if (*p) > 1 {
			Dlacpy('L', toPtr((*p)-1), n, b.Off(1, 0), ldb, v.Off(1, 0), ldv)
		}
		Dorg2r(p, p, toPtr(min(*p, *n)), v, ldv, tau, work, info)
	}

	//     Clean up B
	for j = 1; j <= (*l)-1; j++ {
		for i = j + 1; i <= (*l); i++ {
			b.Set(i-1, j-1, zero)
		}
	}
	if (*p) > (*l) {
		Dlaset('F', toPtr((*p)-(*l)), n, &zero, &zero, b.Off((*l), 0), ldb)
	}

	if wantq {
		//        Set Q = I and Update Q := Q*P
		Dlaset('F', n, n, &zero, &one, q, ldq)
		Dlapmt(forwrd, n, n, q, ldq, iwork)
	}

	if (*p) >= (*l) && (*n) != (*l) {
		//        RQ factorization of (S11 S12): ( S11 S12 ) = ( 0 S12 )*Z
		Dgerq2(l, n, b, ldb, tau, work, info)

		//        Update A := A*Z**T
		Dormr2('R', 'T', m, n, l, b, ldb, tau, a, lda, work, info)

		if wantq {
			//           Update Q := Q*Z**T
			Dormr2('R', 'T', n, n, l, b, ldb, tau, q, ldq, work, info)
		}

		//        Clean up B
		Dlaset('F', l, toPtr((*n)-(*l)), &zero, &zero, b, ldb)
		for j = (*n) - (*l) + 1; j <= (*n); j++ {
			for i = j - (*n) + (*l) + 1; i <= (*l); i++ {
				b.Set(i-1, j-1, zero)
			}
		}

	}

	//     Let              N-L     L
	//                A = ( A11    A12 ) M,
	//
	//     then the following does the complete QR decomposition of A11:
	//
	//              A11 = U*(  0  T12 )*P1**T
	//                      (  0   0  )
	for i = 1; i <= (*n)-(*l); i++ {
		(*iwork)[i-1] = 0
	}
	Dgeqp3(m, toPtr((*n)-(*l)), a, lda, iwork, tau, work, lwork, info)

	//     Determine the effective rank of A11
	(*k) = 0
	for i = 1; i <= min(*m, (*n)-(*l)); i++ {
		if math.Abs(a.Get(i-1, i-1)) > (*tola) {
			(*k) = (*k) + 1
		}
	}

	//     Update A12 := U**T*A12, where A12 = A( 1:M, N-L+1:N )
	Dorm2r('L', 'T', m, l, toPtr(min(*m, (*n)-(*l))), a, lda, tau, a.Off(0, (*n)-(*l)), lda, work, info)

	if wantu {
		//        Copy the details of U, and form U
		Dlaset('F', m, m, &zero, &zero, u, ldu)
		if (*m) > 1 {
			Dlacpy('L', toPtr((*m)-1), toPtr((*n)-(*l)), a.Off(1, 0), lda, u.Off(1, 0), ldu)
		}
		Dorg2r(m, m, toPtr(min(*m, (*n)-(*l))), u, ldu, tau, work, info)
	}

	if wantq {
		//        Update Q( 1:N, 1:N-L )  = Q( 1:N, 1:N-L )*P1
		Dlapmt(forwrd, n, toPtr((*n)-(*l)), q, ldq, iwork)
	}

	//     Clean up A: set the strictly lower triangular part of
	//     A(1:K, 1:K) = 0, and A( K+1:M, 1:N-L ) = 0.
	for j = 1; j <= (*k)-1; j++ {
		for i = j + 1; i <= (*k); i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	if (*m) > (*k) {
		Dlaset('F', toPtr((*m)-(*k)), toPtr((*n)-(*l)), &zero, &zero, a.Off((*k), 0), lda)
	}

	if (*n)-(*l) > (*k) {
		//        RQ factorization of ( T11 T12 ) = ( 0 T12 )*Z1
		Dgerq2(k, toPtr((*n)-(*l)), a, lda, tau, work, info)

		if wantq {
			//           Update Q( 1:N,1:N-L ) = Q( 1:N,1:N-L )*Z1**T
			Dormr2('R', 'T', n, toPtr((*n)-(*l)), k, a, lda, tau, q, ldq, work, info)
		}

		//        Clean up A
		Dlaset('F', k, toPtr((*n)-(*l)-(*k)), &zero, &zero, a, lda)
		for j = (*n) - (*l) - (*k) + 1; j <= (*n)-(*l); j++ {
			for i = j - (*n) + (*l) + (*k) + 1; i <= (*k); i++ {
				a.Set(i-1, j-1, zero)
			}
		}

	}

	if (*m) > (*k) {
		//        QR factorization of A( K+1:M,N-L+1:N )
		Dgeqr2(toPtr((*m)-(*k)), l, a.Off((*k), (*n)-(*l)), lda, tau, work, info)

		if wantu {
			//           Update U(:,K+1:M) := U(:,K+1:M)*U1
			Dorm2r('R', 'N', m, toPtr((*m)-(*k)), toPtr(min((*m)-(*k), *l)), a.Off((*k), (*n)-(*l)), lda, tau, u.Off(0, (*k)), ldu, work, info)
		}

		//        Clean up
		for j = (*n) - (*l) + 1; j <= (*n); j++ {
			for i = j - (*n) + (*k) + (*l) + 1; i <= (*m); i++ {
				a.Set(i-1, j-1, zero)
			}
		}

	}

	work.Set(0, float64(lwkopt))
}
