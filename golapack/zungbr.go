package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zungbr generates one of the complex unitary matrices Q or P**H
// determined by ZGEBRD when reducing a complex matrix A to bidiagonal
// form: A = Q * B * P**H.  Q and P**H are defined as products of
// elementary reflectors H(i) or G(i) respectively.
//
// If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
// is of order M:
// if m >= k, Q = H(1) H(2) . . . H(k) and ZUNGBR returns the first n
// columns of Q, where m >= n >= k;
// if m < k, Q = H(1) H(2) . . . H(m-1) and ZUNGBR returns Q as an
// M-by-M matrix.
//
// If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**H
// is of order N:
// if k < n, P**H = G(k) . . . G(2) G(1) and ZUNGBR returns the first m
// rows of P**H, where n >= m >= k;
// if k >= n, P**H = G(n-1) . . . G(2) G(1) and ZUNGBR returns P**H as
// an N-by-N matrix.
func Zungbr(vect byte, m, n, k *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery, wantq bool
	var one, zero complex128
	var i, iinfo, j, lwkopt, mn int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	wantq = vect == 'Q'
	mn = minint(*m, *n)
	lquery = ((*lwork) == -1)
	if !wantq && vect != 'P' {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -2
	} else if (*n) < 0 || (wantq && ((*n) > (*m) || (*n) < minint(*m, *k))) || (!wantq && ((*m) > (*n) || (*m) < minint(*n, *k))) {
		(*info) = -3
	} else if (*k) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *m) {
		(*info) = -6
	} else if (*lwork) < maxint(1, mn) && !lquery {
		(*info) = -9
	}

	if (*info) == 0 {
		work.Set(0, 1)
		if wantq {
			if (*m) >= (*k) {
				Zungqr(m, n, k, a, lda, tau, work, toPtr(-1), &iinfo)
			} else {
				if (*m) > 1 {
					Zungqr(toPtr((*m)-1), toPtr((*m)-1), toPtr((*m)-1), a.Off(1, 1), lda, tau, work, toPtr(-1), &iinfo)
				}
			}
		} else {
			if (*k) < (*n) {
				Zunglq(m, n, k, a, lda, tau, work, toPtr(-1), &iinfo)
			} else {
				if (*n) > 1 {
					Zunglq(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), a.Off(1, 1), lda, tau, work, toPtr(-1), &iinfo)
				}
			}
		}
		lwkopt = int(work.GetRe(0))
		lwkopt = maxint(lwkopt, mn)
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGBR"), -(*info))
		return
	} else if lquery {
		work.SetRe(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		work.Set(0, 1)
		return
	}

	if wantq {
		//        Form Q, determined by a call to ZGEBRD to reduce an m-by-k
		//        matrix
		if (*m) >= (*k) {
			//           If m >= k, assume m >= n >= k
			Zungqr(m, n, k, a, lda, tau, work, lwork, &iinfo)

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
				Zungqr(toPtr((*m)-1), toPtr((*m)-1), toPtr((*m)-1), a.Off(1, 1), lda, tau, work, lwork, &iinfo)
			}
		}
	} else {
		//        Form P**H, determined by a call to ZGEBRD to reduce a k-by-n
		//        matrix
		if (*k) < (*n) {
			//           If k < n, assume k <= m <= n
			Zunglq(m, n, k, a, lda, tau, work, lwork, &iinfo)

		} else {
			//           If k >= n, assume m = n
			//
			//           Shift the vectors which define the elementary reflectors one
			//           row downward, and set the first row and column of P**H to
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
				//              Form P**H(2:n,2:n)
				Zunglq(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), a.Off(1, 1), lda, tau, work, lwork, &iinfo)
			}
		}
	}
	work.SetRe(0, float64(lwkopt))
}
