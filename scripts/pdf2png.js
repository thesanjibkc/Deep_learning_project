ObjC.import('PDFKit'); ObjC.import('AppKit');
function run(argv){
  var doc=$.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  var scale=argv.length>2?parseFloat(argv[2]):2, out=[];
  for(var i=0;i<doc.pageCount;i++){
    var page=doc.pageAtIndex(i), r=page.boundsForBox($.kPDFDisplayBoxMediaBox);
    var w=Math.round(r.size.width*scale), h=Math.round(r.size.height*scale);
    var rep=$.NSBitmapImageRep.alloc.initWithBitmapDataPlanesPixelsWidePixelsHighBitsPerSampleSamplesPerPixelHasAlphaIsPlanarColorSpaceNameBytesPerRowBitsPerPixel(null,w,h,8,4,true,false,$.NSDeviceRGBColorSpace,0,0);
    $.NSGraphicsContext.saveGraphicsState;
    var ctx=$.NSGraphicsContext.graphicsContextWithBitmapImageRep(rep);
    $.NSGraphicsContext.setCurrentContext(ctx);
    $.NSColor.whiteColor.setFill; $.NSRectFill($.NSMakeRect(0,0,w,h));
    var t=$.NSAffineTransform.transform; t.scaleBy(scale); t.concat;
    page.drawWithBoxToContext($.kPDFDisplayBoxMediaBox, ctx.CGContext);
    $.NSGraphicsContext.restoreGraphicsState;
    var p=argv[1]+'/page-'+(i<9?'0':'')+(i+1)+'.png';
    rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG,$()).writeToFileAtomically(p,true);
    out.push(p);
  }
  return out.join('\n');
}
