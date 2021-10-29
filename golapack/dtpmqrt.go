package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtpmqrt applies a real orthogonal matrix Q obtained from a
// "triangular-pentagonal" real block reflector H to a general
// real matrix C, which consists of two blocks A and B.
func Dtpmqrt(side mat.MatSide, trans mat.MatTrans, m, n, k, l, nb int, v, t, a, b *mat.Matrix, work *mat.Vector) (err error) {
	var left, notran, right, tran bool
	var i, ib, kf, lb, ldaq, ldvq, mb int

	//     .. Test the input arguments ..
	left = side == Left
	right = side == Right
	tran = trans == Trans
	notran = trans == NoTrans

	if left {
		ldvq = max(1, m)
		ldaq = max(1, k)
	} else if right {
		ldvq = max(1, n)
		ldaq = max(1, m)
	}
	if !left && !right {
		err = fmt.Errorf("!left && !right: side=%s", side)
	} else if !tran && !notran {
		err = fmt.Errorf("!tran && !notran: trans=%s", trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if k < 0 {
		err = fmt.Errorf("k < 0: k=%v", k)
	} else if l < 0 || l > k {
		err = fmt.Errorf("l < 0 || l > k: l=%v, k=%v", l, k)
	} else if nb < 1 || (nb > k && k > 0) {
		err = fmt.Errorf("nb < 1 || (nb > k && k > 0): nb=%v, k=%v", nb, k)
	} else if v.Rows < ldvq {
		err = fmt.Errorf("v.Rows < ldvq: v.Rows=%v, ldvq=%v", v.Rows, ldvq)
	} else if t.Rows < nb {
		err = fmt.Errorf("t.Rows < nb: t.Rows=%v, nb=%v", t.Rows, nb)
	} else if a.Rows < ldaq {
		err = fmt.Errorf("a.Rows < ldaq: a.Rows=%v, ldaq=%v", a.Rows, ldaq)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	}

	if err != nil {
		gltest.Xerbla2("Dtpmqrt", err)
		return
	}

	//     .. Quick return if possible ..
	if m == 0 || n == 0 || k == 0 {
		return
	}

	if left && tran {

		for i = 1; i <= k; i += nb {
			ib = min(nb, k-i+1)
			mb = min(m-l+i+ib-1, m)
			if i >= l {
				lb = 0
			} else {
				lb = mb - m + l - i + 1
			}
			Dtprfb(Left, Trans, 'F', 'C', mb, n, ib, lb, v.Off(0, i-1), t.Off(0, i-1), a.Off(i-1, 0), b, work.Matrix(ib, opts))
		}

	} else if right && notran {

		for i = 1; i <= k; i += nb {
			ib = min(nb, k-i+1)
			mb = min(n-l+i+ib-1, n)
			if i >= l {
				lb = 0
			} else {
				lb = mb - n + l - i + 1
			}
			Dtprfb(Right, NoTrans, 'F', 'C', m, mb, ib, lb, v.Off(0, i-1), t.Off(0, i-1), a.Off(0, i-1), b, work.Matrix(m, opts))
		}

	} else if left && notran {

		kf = ((k-1)/nb)*nb + 1
		for i = kf; i >= 1; i -= nb {
			ib = min(nb, k-i+1)
			mb = min(m-l+i+ib-1, m)
			if i >= l {
				lb = 0
			} else {
				lb = mb - m + l - i + 1
			}
			Dtprfb(Left, NoTrans, 'F', 'C', mb, n, ib, lb, v.Off(0, i-1), t.Off(0, i-1), a.Off(i-1, 0), b, work.Matrix(ib, opts))
		}

	} else if right && tran {

		kf = ((k-1)/nb)*nb + 1
		for i = kf; i >= 1; i -= nb {
			ib = min(nb, k-i+1)
			mb = min(n-l+i+ib-1, n)
			if i >= l {
				lb = 0
			} else {
				lb = mb - n + l - i + 1
			}
			Dtprfb(Right, Trans, 'F', 'C', m, mb, ib, lb, v.Off(0, i-1), t.Off(0, i-1), a.Off(0, i-1), b, work.Matrix(m, opts))
		}

	}

	return
}
