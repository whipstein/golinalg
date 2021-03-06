package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgebak forms the right or left eigenvectors of a real general matrix
// by backward transformation on the computed eigenvectors of the
// balanced matrix output by DGEBAL.
func Dgebak(job byte, side mat.MatSide, n, ilo, ihi int, scale *mat.Vector, m int, v *mat.Matrix) (err error) {
	var leftv, rightv bool
	var one, s float64
	var i, ii, k int

	one = 1.0

	//     Decode and Test the input parameters
	rightv = side == Right
	leftv = side == Left

	if job != 'N' && job != 'P' && job != 'S' && job != 'B' {
		err = fmt.Errorf("job != 'N' && job != 'P' && job != 'S' && job != 'B': job='%c'", job)
	} else if !rightv && !leftv {
		err = fmt.Errorf("!rightv && !leftv: side=%s", side)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 || ilo > max(1, n) {
		err = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=%v, n=%v", ilo, n)
	} else if ihi < min(ilo, n) || ihi > n {
		err = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=%v, ihi=%v, n=%v", ilo, ihi, n)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if v.Rows < max(1, n) {
		err = fmt.Errorf("v.Rows < max(1, n): v.Rows=%v, n=%v", v.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgebak", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}
	if m == 0 {
		return
	}
	if job == 'N' {
		return
	}

	if ilo == ihi {
		goto label30
	}

	//     Backward balance
	if job == 'S' || job == 'B' {

		if rightv {
			for i = ilo; i <= ihi; i++ {
				s = scale.Get(i - 1)
				v.Off(i-1, 0).Vector().Scal(m, s, v.Rows)
			}
		}

		if leftv {
			for i = ilo; i <= ihi; i++ {
				s = one / scale.Get(i-1)
				v.Off(i-1, 0).Vector().Scal(m, s, v.Rows)
			}
		}

	}

	//     Backward permutation
	//
	//     For  I = ILO-1 step -1 until 1,
	//              IHI+1 step 1 until N do --
label30:
	;
	if job == 'P' || job == 'B' {
		if rightv {
			for ii = 1; ii <= n; ii++ {
				i = ii
				if i >= ilo && i <= ihi {
					goto label40
				}
				if i < ilo {
					i = ilo - ii
				}
				k = int(scale.Get(i - 1))
				if k == i {
					goto label40
				}
				v.Off(k-1, 0).Vector().Swap(m, v.Off(i-1, 0).Vector(), v.Rows, v.Rows)
			label40:
			}
		}

		if leftv {
			for ii = 1; ii <= n; ii++ {
				i = ii
				if i >= ilo && i <= ihi {
					goto label50
				}
				if i < ilo {
					i = ilo - ii
				}
				k = int(scale.Get(i - 1))
				if k == i {
					goto label50
				}
				v.Off(k-1, 0).Vector().Swap(m, v.Off(i-1, 0).Vector(), v.Rows, v.Rows)
			label50:
			}
		}
	}

	return
}
