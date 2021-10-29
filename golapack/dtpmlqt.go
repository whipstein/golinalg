package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtpmlqt applies a real orthogonal matrix Q obtained from a
// "triangular-pentagonal" real block reflector H to a general
// real matrix C, which consists of two blocks A and B.
func Dtpmlqt(side mat.MatSide, trans mat.MatTrans, m, n, k, l, mb int, v, t, a, b *mat.Matrix, work *mat.Vector) (err error) {
	var left, notran, right, tran bool
	var i, ib, kf, lb, ldaq, nb int

	//     .. Test the input arguments ..
	left = side == Left
	right = side == Right
	tran = trans == Trans
	notran = trans == NoTrans

	if left {
		ldaq = max(1, k)
	} else if right {
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
	} else if mb < 1 || (mb > k && k > 0) {
		err = fmt.Errorf("mb < 1 || (mb > k && k > 0): mb=%v, k=%v", mb, k)
	} else if v.Rows < k {
		err = fmt.Errorf("v.Rows < k: v.Rows=%v, k=%v", v.Rows, k)
	} else if t.Rows < mb {
		err = fmt.Errorf("t.Rows < mb: t.Rows=%v, mb=%v", t.Rows, mb)
	} else if a.Rows < ldaq {
		err = fmt.Errorf("a.Rows < ldaq: a.Rows=%v, ldaq=%v", a.Rows, ldaq)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	}

	if err != nil {
		gltest.Xerbla2("Dtpmlqt", err)
		return
	}

	//     .. Quick return if possible ..
	if m == 0 || n == 0 || k == 0 {
		return
	}

	if left && notran {

		for i = 1; i <= k; i += mb {
			ib = min(mb, k-i+1)
			nb = min(m-l+i+ib-1, m)
			if i >= l {
				lb = 0
			} else {
				lb = 0
			}
			Dtprfb(Left, Trans, 'F', 'R', nb, n, ib, lb, v.Off(i-1, 0), t.Off(0, i-1), a.Off(i-1, 0), b, work.Matrix(ib, opts))
		}

	} else if right && tran {

		for i = 1; i <= k; i += mb {
			ib = min(mb, k-i+1)
			nb = min(n-l+i+ib-1, n)
			if i >= l {
				lb = 0
			} else {
				lb = nb - n + l - i + 1
			}
			Dtprfb(Right, NoTrans, 'F', 'R', m, nb, ib, lb, v.Off(i-1, 0), t.Off(0, i-1), a.Off(0, i-1), b, work.Matrix(m, opts))
		}

	} else if left && tran {

		kf = ((k-1)/mb)*mb + 1
		for i = kf; i >= 1; i -= mb {
			ib = min(mb, k-i+1)
			nb = min(m-l+i+ib-1, m)
			if i >= l {
				lb = 0
			} else {
				lb = 0
			}
			Dtprfb(Left, NoTrans, 'F', 'R', nb, n, ib, lb, v.Off(i-1, 0), t.Off(0, i-1), a.Off(i-1, 0), b, work.Matrix(ib, opts))
		}

	} else if right && notran {

		kf = ((k-1)/mb)*mb + 1
		for i = kf; i >= 1; i -= mb {
			ib = min(mb, k-i+1)
			nb = min(n-l+i+ib-1, n)
			if i >= l {
				lb = 0
			} else {
				lb = nb - n + l - i + 1
			}
			Dtprfb(Right, Trans, 'F', 'R', m, nb, ib, lb, v.Off(i-1, 0), t.Off(0, i-1), a.Off(0, i-1), b, work.Matrix(m, opts))
		}

	}

	return
}
