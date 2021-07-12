package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsterf computes all eigenvalues of a symmetric tridiagonal matrix
// using the Pal-Walker-Kahan variant of the QL or QR algorithm.
func Dsterf(n *int, d, e *mat.Vector, info *int) {
	var alpha, anorm, bb, c, eps, eps2, gamma, oldc, oldgam, one, p, r, rt1, rt2, rte, s, safmax, safmin, sigma, ssfmax, ssfmin, three, two, zero float64
	var i, iscale, jtot, l, l1, lend, lendsv, lsv, m, maxit, nmaxit int

	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0
	maxit = 30

	//     Test the input parameters.
	(*info) = 0

	//     Quick return if possible
	if (*n) < 0 {
		(*info) = -1
		gltest.Xerbla([]byte("DSTERF"), -(*info))
		return
	}
	if (*n) <= 1 {
		return
	}

	//     Determine the unit roundoff for this environment.
	eps = Dlamch(Epsilon)
	eps2 = math.Pow(eps, 2)
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	ssfmax = math.Sqrt(safmax) / three
	ssfmin = math.Sqrt(safmin) / eps2
	// rmax = Dlamch(Overflow)

	//     Compute the eigenvalues of the tridiagonal matrix.
	nmaxit = (*n) * maxit
	sigma = zero
	jtot = 0

	//     Determine where the matrix splits and choose QL or QR iteration
	//     for each block, according to whether top or bottom diagonal
	//     element is smaller.
	l1 = 1

label10:
	;
	if l1 > (*n) {
		goto label170
	}
	if l1 > 1 {
		e.Set(l1-1-1, zero)
	}
	for m = l1; m <= (*n)-1; m++ {
		if math.Abs(e.Get(m-1)) <= (math.Sqrt(math.Abs(d.Get(m-1)))*math.Sqrt(math.Abs(d.Get(m))))*eps {
			e.Set(m-1, zero)
			goto label30
		}
	}
	m = (*n)

label30:
	;
	l = l1
	lsv = l
	lend = m
	lendsv = lend
	l1 = m + 1
	if lend == l {
		goto label10
	}

	//     Scale submatrix in rows and columns L to LEND
	anorm = Dlanst('M', toPtr(lend-l+1), d.Off(l-1), e.Off(l-1))
	iscale = 0
	if anorm == zero {
		goto label10
	}
	if anorm > ssfmax {
		iscale = 1
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmax, toPtr(lend-l+1), func() *int { y := 1; return &y }(), d.MatrixOff(l-1, *n, opts), n, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmax, toPtr(lend-l), func() *int { y := 1; return &y }(), e.MatrixOff(l-1, *n, opts), n, info)
	} else if anorm < ssfmin {
		iscale = 2
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmin, toPtr(lend-l+1), func() *int { y := 1; return &y }(), d.MatrixOff(l-1, *n, opts), n, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmin, toPtr(lend-l), func() *int { y := 1; return &y }(), e.MatrixOff(l-1, *n, opts), n, info)
	}

	for i = l; i <= lend-1; i++ {
		e.Set(i-1, math.Pow(e.Get(i-1), 2))
	}

	//     Choose between QL and QR iteration
	if math.Abs(d.Get(lend-1)) < math.Abs(d.Get(l-1)) {
		lend = lsv
		l = lendsv
	}

	if lend >= l {
		//        QL Iteration
		//
		//        Look for small subdiagonal element.
	label50:
		;
		if l != lend {
			for m = l; m <= lend-1; m++ {
				if math.Abs(e.Get(m-1)) <= eps2*math.Abs(d.Get(m-1)*d.Get(m)) {
					goto label70
				}
			}
		}
		m = lend

	label70:
		;
		if m < lend {
			e.Set(m-1, zero)
		}
		p = d.Get(l - 1)
		if m == l {
			goto label90
		}

		//        If remaining matrix is 2 by 2, use DLAE2 to compute its
		//        eigenvalues.
		if m == l+1 {
			rte = math.Sqrt(e.Get(l - 1))
			Dlae2(d.GetPtr(l-1), &rte, d.GetPtr(l), &rt1, &rt2)
			d.Set(l-1, rt1)
			d.Set(l, rt2)
			e.Set(l-1, zero)
			l = l + 2
			if l <= lend {
				goto label50
			}
			goto label150
		}

		if jtot == nmaxit {
			goto label150
		}
		jtot = jtot + 1

		//        Form shift.
		rte = math.Sqrt(e.Get(l - 1))
		sigma = (d.Get(l) - p) / (two * rte)
		r = Dlapy2(&sigma, &one)
		sigma = p - (rte / (sigma + math.Copysign(r, sigma)))

		c = one
		s = zero
		gamma = d.Get(m-1) - sigma
		p = gamma * gamma

		//        Inner loop
		for i = m - 1; i >= l; i-- {
			bb = e.Get(i - 1)
			r = p + bb
			if i != m-1 {
				e.Set(i, s*r)
			}
			oldc = c
			c = p / r
			s = bb / r
			oldgam = gamma
			alpha = d.Get(i - 1)
			gamma = c*(alpha-sigma) - s*oldgam
			d.Set(i, oldgam+(alpha-gamma))
			if c != zero {
				p = (gamma * gamma) / c
			} else {
				p = oldc * bb
			}
		}

		e.Set(l-1, s*p)
		d.Set(l-1, sigma+gamma)
		goto label50

		//        Eigenvalue found.
	label90:
		;
		d.Set(l-1, p)

		l = l + 1
		if l <= lend {
			goto label50
		}
		goto label150

	} else {
		//        QR Iteration
		//
		//        Look for small superdiagonal element.
	label100:
		;
		for m = l; m >= lend+1; m-- {
			if math.Abs(e.Get(m-1-1)) <= eps2*math.Abs(d.Get(m-1)*d.Get(m-1-1)) {
				goto label120
			}
		}
		m = lend

	label120:
		;
		if m > lend {
			e.Set(m-1-1, zero)
		}
		p = d.Get(l - 1)
		if m == l {
			goto label140
		}

		//        If remaining matrix is 2 by 2, use DLAE2 to compute its
		//        eigenvalues.
		if m == l-1 {
			rte = math.Sqrt(e.Get(l - 1 - 1))
			Dlae2(d.GetPtr(l-1), &rte, d.GetPtr(l-1-1), &rt1, &rt2)
			d.Set(l-1, rt1)
			d.Set(l-1-1, rt2)
			e.Set(l-1-1, zero)
			l = l - 2
			if l >= lend {
				goto label100
			}
			goto label150
		}

		if jtot == nmaxit {
			goto label150
		}
		jtot = jtot + 1

		//        Form shift.
		rte = math.Sqrt(e.Get(l - 1 - 1))
		sigma = (d.Get(l-1-1) - p) / (two * rte)
		r = Dlapy2(&sigma, &one)
		sigma = p - (rte / (sigma + math.Copysign(r, sigma)))

		c = one
		s = zero
		gamma = d.Get(m-1) - sigma
		p = gamma * gamma

		//        Inner loop
		for i = m; i <= l-1; i++ {
			bb = e.Get(i - 1)
			r = p + bb
			if i != m {
				e.Set(i-1-1, s*r)
			}
			oldc = c
			c = p / r
			s = bb / r
			oldgam = gamma
			alpha = d.Get(i + 1 - 1)
			gamma = c*(alpha-sigma) - s*oldgam
			d.Set(i-1, oldgam+(alpha-gamma))
			if c != zero {
				p = (gamma * gamma) / c
			} else {
				p = oldc * bb
			}
		}

		e.Set(l-1-1, s*p)
		d.Set(l-1, sigma+gamma)
		goto label100

		//        Eigenvalue found.
	label140:
		;
		d.Set(l-1, p)

		l = l - 1
		if l >= lend {
			goto label100
		}
		goto label150

	}

	//     Undo scaling if necessary
label150:
	;
	if iscale == 1 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ssfmax, &anorm, toPtr(lendsv-lsv+1), func() *int { y := 1; return &y }(), d.MatrixOff(lsv-1, *n, opts), n, info)
	}
	if iscale == 2 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ssfmin, &anorm, toPtr(lendsv-lsv+1), func() *int { y := 1; return &y }(), d.MatrixOff(lsv-1, *n, opts), n, info)
	}

	//     Check for no convergence to an eigenvalue after a total
	//     of N*MAXIT iterations.
	if jtot < nmaxit {
		goto label10
	}
	for i = 1; i <= (*n)-1; i++ {
		if e.Get(i-1) != zero {
			(*info) = (*info) + 1
		}
	}
	goto label180

	//     Sort eigenvalues in increasing order.
label170:
	;
	Dlasrt('I', n, d, info)

label180:
}
