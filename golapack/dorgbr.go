package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorgbr generates one of the real orthogonal matrices Q or P**T
// determined by DGEBRD when reducing a real matrix A to bidiagonal
// form: A = Q * B * P**T.  Q and P**T are defined as products of
// elementary reflectors H(i) or G(i) respectively.
//
// If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
// is of order M:
// if m >= k, Q = H(1) H(2) . . . H(k) and DORGBR returns the first n
// columns of Q, where m >= n >= k;
// if m < k, Q = H(1) H(2) . . . H(m-1) and DORGBR returns Q as an
// M-by-M matrix.
//
// If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**T
// is of order N:
// if k < n, P**T = G(k) . . . G(2) G(1) and DORGBR returns the first m
// rows of P**T, where n >= m >= k;
// if k >= n, P**T = G(n-1) . . . G(2) G(1) and DORGBR returns P**T as
// an N-by-N matrix.
func Dorgbr(vect byte, m, n, k *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, lwork, info *int) {
	var lquery, wantq bool
	var one, zero float64
	var i, iinfo, j, lwkopt, mn int

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	(*info) = 0
	wantq = vect == 'Q'
	mn = min(*m, *n)
	lquery = ((*lwork) == -1)
	if !wantq && vect != 'P' {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -2
	} else if (*n) < 0 || (wantq && ((*n) > (*m) || (*n) < min(*m, *k))) || (!wantq && ((*m) > (*n) || (*m) < min(*n, *k))) {
		(*info) = -3
	} else if (*k) < 0 {
		(*info) = -4
	} else if (*lda) < max(1, *m) {
		(*info) = -6
	} else if (*lwork) < max(1, mn) && !lquery {
		(*info) = -9
	}

	if (*info) == 0 {
		work.Set(0, 1)
		if wantq {
			if (*m) >= (*k) {
				Dorgqr(m, n, k, a, lda, tau, work, toPtr(-1), &iinfo)
			} else {
				if (*m) > 1 {
					Dorgqr(toPtr((*m)-1), toPtr((*m)-1), toPtr((*m)-1), a.Off(1, 1), lda, tau, work, toPtr(-1), &iinfo)
				}
			}
		} else {
			if (*k) < (*n) {
				Dorglq(m, n, k, a, lda, tau, work, toPtr(-1), &iinfo)
			} else {
				if (*n) > 1 {
					Dorglq(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), a.Off(1, 1), lda, tau, work, toPtr(-1), &iinfo)
				}
			}
		}
		lwkopt = int(work.Get(0))
		lwkopt = max(lwkopt, mn)
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DORGBR"), -(*info))
		return
	} else if lquery {
		work.Set(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		work.Set(0, 1)
		return
	}

	if wantq {
		//        Form Q, determined by a call to DGEBRD to reduce an m-by-k
		//        matrix
		if (*m) >= (*k) {
			//           If m >= k, assume m >= n >= k
			Dorgqr(m, n, k, a, lda, tau, work, lwork, &iinfo)

		} else {
			//           If m < k, assume m = n
			//
			//           Shift the vectors which define the elementary reflectors one
			//           column to the right, and set the first row and column of Q
			//           to those of the unit matrix
			for j = (*m); j >= 2; j-- {
				a.Set(0, j-1, zero)
				for i = j + 1; i <= (*m); i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1-1))
				}
			}
			a.Set(0, 0, one)
			for i = 2; i <= (*m); i++ {
				a.Set(i-1, 0, zero)
			}
			if (*m) > 1 {
				//              Form Q(2:m,2:m)
				Dorgqr(toPtr((*m)-1), toPtr((*m)-1), toPtr((*m)-1), a.Off(1, 1), lda, tau, work, lwork, &iinfo)
			}
		}
	} else {
		//        Form P**T, determined by a call to DGEBRD to reduce a k-by-n
		//        matrix
		if (*k) < (*n) {
			//           If k < n, assume k <= m <= n
			Dorglq(m, n, k, a, lda, tau, work, lwork, &iinfo)

		} else {
			//           If k >= n, assume m = n
			//
			//           Shift the vectors which define the elementary reflectors one
			//           row downward, and set the first row and column of P**T to
			//           those of the unit matrix
			a.Set(0, 0, one)
			for i = 2; i <= (*n); i++ {
				a.Set(i-1, 0, zero)
			}
			for j = 2; j <= (*n); j++ {
				for i = j - 1; i >= 2; i-- {
					a.Set(i-1, j-1, a.Get(i-1-1, j-1))
				}
				a.Set(0, j-1, zero)
			}
			if (*n) > 1 {
				//              Form P**T(2:n,2:n)
				Dorglq(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), a.Off(1, 1), lda, tau, work, lwork, &iinfo)
			}
		}
	}
	work.Set(0, float64(lwkopt))
}
