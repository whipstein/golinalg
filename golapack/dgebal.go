package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgebal balances a general real matrix A.  This involves, first,
// permuting A by a similarity transformation to isolate eigenvalues
// in the first 1 to ILO-1 and last IHI+1 to N elements on the
// diagonal; and second, applying a diagonal similarity transformation
// to rows and columns ILO to IHI to make the rows and columns as
// close in norm as possible.  Both steps are optional.
//
// Balancing may reduce the 1-norm of the matrix, and improve the
// accuracy of the computed eigenvalues and/or eigenvectors.
func Dgebal(job byte, n int, a *mat.Matrix, scale *mat.Vector) (ilo, ihi int, err error) {
	var noconv bool
	var c, ca, f, factor, g, one, r, ra, s, sclfac, sfmax1, sfmax2, sfmin1, sfmin2, zero float64
	var i, ica, iexc, ira, j, k, l, m int

	zero = 0.0
	one = 1.0
	sclfac = 2.0
	factor = 0.95

	//     Test the input parameters
	if job != 'N' && job != 'P' && job != 'S' && job != 'B' {
		err = fmt.Errorf("job != 'N' && job != 'P' && job != 'S' && job != 'B': job='%c'", job)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgebal", err)
		return
	}

	k = 1
	l = n

	if n == 0 {
		goto label210
	}

	if job == 'N' {
		for i = 1; i <= n; i++ {
			scale.Set(i-1, one)
		}
		goto label210
	}

	if job == 'S' {
		goto label120
	}

	//     Permutation to isolate eigenvalues if possible
	goto label50

	//     Row and column exchange.
label20:
	;
	scale.Set(m-1, float64(j))
	if j == m {
		goto label30
	}

	goblas.Dswap(l, a.Vector(0, j-1, 1), a.Vector(0, m-1, 1))
	goblas.Dswap(n-k+1, a.Vector(j-1, k-1), a.Vector(m-1, k-1))

label30:
	;
	switch iexc {
	case 1:
		goto label40
	case 2:
		goto label80
	}

	//     Search for rows isolating an eigenvalue and push them down.
label40:
	;
	if l == 1 {
		goto label210
	}
	l = l - 1

label50:
	;
	for j = l; j >= 1; j-- {

		for i = 1; i <= l; i++ {
			if i == j {
				goto label60
			}
			if a.Get(j-1, i-1) != zero {
				goto label70
			}
		label60:
		}

		m = l
		iexc = 1
		goto label20
	label70:
	}

	goto label90

	//     Search for columns isolating an eigenvalue and push them left.
label80:
	;
	k = k + 1

label90:
	;
	for j = k; j <= l; j++ {

		for i = k; i <= l; i++ {
			if i == j {
				goto label100
			}
			if a.Get(i-1, j-1) != zero {
				goto label110
			}
		label100:
		}

		m = k
		iexc = 2
		goto label20
	label110:
	}

label120:
	;
	for i = k; i <= l; i++ {
		scale.Set(i-1, one)
	}

	if job == 'P' {
		goto label210
	}

	//     Balance the submatrix in rows K to L.
	//
	//     Iterative loop for norm reduction
	sfmin1 = Dlamch(SafeMinimum) / Dlamch(Precision)
	sfmax1 = one / sfmin1
	sfmin2 = sfmin1 * sclfac
	sfmax2 = one / sfmin2

label140:
	;
	noconv = false

	for i = k; i <= l; i++ {

		c = goblas.Dnrm2(l-k+1, a.Vector(k-1, i-1, 1))
		r = goblas.Dnrm2(l-k+1, a.Vector(i-1, k-1))
		ica = goblas.Idamax(l, a.Vector(0, i-1, 1))
		ca = math.Abs(a.Get(ica-1, i-1))
		ira = goblas.Idamax(n-k+1, a.Vector(i-1, k-1))
		ra = math.Abs(a.Get(i-1, ira+k-1-1))

		//        Guard against zero C or R due to underflow.
		if c == zero || r == zero {
			goto label200
		}
		g = r / sclfac
		f = one
		s = c + r
	label160:
		;
		if c >= g || math.Max(f, math.Max(c, ca)) >= sfmax2 || math.Min(r, math.Min(g, ra)) <= sfmin2 {
			goto label170
		}
		if Disnan(int(c + f + ca + r + g + ra)) {
			//           Exit if NaN to avoid infinite loop
			err = fmt.Errorf("Disnan(int(c + f + ca + r + g + ra)): c=%v, f=%v, ca=%v, r=%v, g=%v, ra=%v", c, f, ca, r, g, ra)
			gltest.Xerbla2("Dgebal", err)
			return
		}
		f = f * sclfac
		c = c * sclfac
		ca = ca * sclfac
		r = r / sclfac
		g = g / sclfac
		ra = ra / sclfac
		goto label160

	label170:
		;
		g = c / sclfac
	label180:
		;
		if g < r || math.Max(r, ra) >= sfmax2 || math.Min(f, math.Min(c, math.Min(g, ca))) <= sfmin2 {
			goto label190
		}
		f = f / sclfac
		c = c / sclfac
		g = g / sclfac
		ca = ca / sclfac
		r = r * sclfac
		ra = ra * sclfac
		goto label180

		//        Now balance.
	label190:
		;
		if (c + r) >= factor*s {
			goto label200
		}
		if f < one && scale.Get(i-1) < one {
			if f*scale.Get(i-1) <= sfmin1 {
				goto label200
			}
		}
		if f > one && scale.Get(i-1) > one {
			if scale.Get(i-1) >= sfmax1/f {
				goto label200
			}
		}
		g = one / f
		scale.Set(i-1, scale.Get(i-1)*f)
		noconv = true

		goblas.Dscal(n-k+1, g, a.Vector(i-1, k-1))
		goblas.Dscal(l, f, a.Vector(0, i-1, 1))

	label200:
	}

	if noconv {
		goto label140
	}

label210:
	;
	ilo = k
	ihi = l

	return
}
