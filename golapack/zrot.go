package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zrot applies a plane rotation, where the cos (C) is real and the
// sin (S) is complex, and the vectors CX and CY are complex.
func Zrot(n int, cx, cy *mat.CVector, c float64, s complex128) {
	var stemp complex128
	var i, ix, iy int

	if n <= 0 {
		return
	}
	if cx.Inc == 1 && cy.Inc == 1 {
		goto label20
	}

	//     Code for unequal increments or equal increments not equal to 1
	ix = 1
	iy = 1
	if cx.Inc < 0 {
		ix = (-n+1)*cx.Inc + 1
	}
	if cy.Inc < 0 {
		iy = (-n+1)*cy.Inc + 1
	}
	for i = 1; i <= n; i++ {
		stemp = complex(c, 0)*cx.Get(ix-1) + s*cy.Get(iy-1)
		cy.Set(iy-1, complex(c, 0)*cy.Get(iy-1)-cmplx.Conj(s)*cx.Get(ix-1))
		cx.Set(ix-1, stemp)
		ix = ix + cx.Inc
		iy = iy + cy.Inc
	}
	return

	//     Code for both increments equal to 1
label20:
	;
	for i = 1; i <= n; i++ {
		stemp = complex(c, 0)*cx.Get(i-1) + s*cy.Get(i-1)
		cy.Set(i-1, complex(c, 0)*cy.Get(i-1)-cmplx.Conj(s)*cx.Get(i-1))
		cx.Set(i-1, stemp)
	}
}
