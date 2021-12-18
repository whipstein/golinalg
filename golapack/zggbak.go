package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggbak forms the right or left eigenvectors of a complex generalized
// eigenvalue problem A*x = lambda*B*x, by backward transformation on
// the computed eigenvectors of the balanced pair of matrices output by
// ZGGBAL.
func Zggbak(job byte, side mat.MatSide, n, ilo, ihi int, lscale, rscale *mat.Vector, m int, v *mat.CMatrix) (err error) {
	var leftv, rightv bool
	var i, k int

	//     Test the input parameters
	rightv = side == Right
	leftv = side == Left

	if job != 'N' && job != 'P' && job != 'S' && job != 'B' {
		err = fmt.Errorf("job != 'N' && job != 'P' && job != 'S' && job != 'B': job='%c'", job)
	} else if !rightv && !leftv {
		err = fmt.Errorf("!rightv && !leftv: side=%s", side)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 {
		err = fmt.Errorf("ilo < 1: ilo=%v", ilo)
	} else if n == 0 && ihi == 0 && ilo != 1 {
		err = fmt.Errorf("n == 0 && ihi == 0 && ilo != 1: n=%v, ilo=%v, ihi=%v", n, ilo, ihi)
	} else if n > 0 && (ihi < ilo || ihi > max(1, n)) {
		err = fmt.Errorf("n > 0 && (ihi < ilo || ihi > max(1, n)): n=%v, ilo=%v, ihi=%v", n, ilo, ihi)
	} else if n == 0 && ilo == 1 && ihi != 0 {
		err = fmt.Errorf("n == 0 && ilo == 1 && ihi != 0: n=%v, ilo=%v, ihi=%v", n, ilo, ihi)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if v.Rows < max(1, n) {
		err = fmt.Errorf("v.Rows < max(1, n): v.Rows=%v, n=%v", v.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zggbak", err)
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
		//        Backward transformation on right eigenvectors
		if rightv {
			for i = ilo; i <= ihi; i++ {
				v.Off(i-1, 0).CVector().Dscal(m, rscale.Get(i-1), v.Rows)
			}
		}

		//        Backward transformation on left eigenvectors
		if leftv {
			for i = ilo; i <= ihi; i++ {
				v.Off(i-1, 0).CVector().Dscal(m, lscale.Get(i-1), v.Rows)
			}
		}
	}

	//     Backward permutation
label30:
	;
	if job == 'P' || job == 'B' {
		//        Backward permutation on right eigenvectors
		if rightv {
			if ilo == 1 {
				goto label50
			}
			for i = ilo - 1; i >= 1; i-- {
				k = int(rscale.Get(i - 1))
				if k == i {
					goto label40
				}
				v.Off(k-1, 0).CVector().Swap(m, v.Off(i-1, 0).CVector(), v.Rows, v.Rows)
			label40:
			}

		label50:
			;
			if ihi == n {
				goto label70
			}
			for i = ihi + 1; i <= n; i++ {
				k = int(rscale.Get(i - 1))
				if k == i {
					goto label60
				}
				v.Off(k-1, 0).CVector().Swap(m, v.Off(i-1, 0).CVector(), v.Rows, v.Rows)
			label60:
			}
		}

		//        Backward permutation on left eigenvectors
	label70:
		;
		if leftv {
			if ilo == 1 {
				goto label90
			}
			for i = ilo - 1; i >= 1; i-- {
				k = int(lscale.Get(i - 1))
				if k == i {
					goto label80
				}
				v.Off(k-1, 0).CVector().Swap(m, v.Off(i-1, 0).CVector(), v.Rows, v.Rows)
			label80:
			}

		label90:
			;
			if ihi == n {
				return
			}
			for i = ihi + 1; i <= n; i++ {
				k = int(lscale.Get(i - 1))
				if k == i {
					goto label100
				}
				v.Off(k-1, 0).CVector().Swap(m, v.Off(i-1, 0).CVector(), v.Rows, v.Rows)
			label100:
			}
		}
	}

	return
}
